//! Bounded scratch sorting for the M4 owner migration.
//!
//! Owner metadata is emitted from concurrent maps in arbitrary order, while
//! direct-address pages and forward indices must be built in key order.  This
//! sorter writes bounded in-memory chunks, then performs fan-in-limited
//! streaming run merges.  At no point is the complete owner key set resident.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

use thiserror::Error;

const RUN_MAGIC: &[u8; 8] = b"AGORS001";
const RUN_HEADER_BYTES: u64 = 16;
const RECORD_PREFIX_BYTES: u64 = 4;
const MERGE_FAN_IN: usize = 16;
/// Bound for one malicious/corrupt metadata record.
pub const OWNER_REWRITE_MAX_RECORD_BYTES: usize = 16 * 1024 * 1024;
/// Default per-sorter resident chunk bound.
pub const OWNER_REWRITE_RUN_BUFFER_BYTES: usize = 8 * 1024 * 1024;
/// Shared scratch ceiling across every owner class during one rewrite.
pub const OWNER_REWRITE_SCRATCH_CAP_BYTES: u64 = 12 * 1024 * 1024 * 1024;

/// Typed external-sort failure.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum OwnerRewriteError {
    /// Filesystem operation failed.
    #[error(transparent)]
    Io(#[from] std::io::Error),
    /// Scratch run is malformed.
    #[error("owner rewrite run is corrupt: {0}")]
    Corrupt(String),
    /// One record exceeds the explicit per-record limit.
    #[error("owner rewrite record has {len} bytes, max is {max}")]
    RecordTooLarge {
        /// Requested bytes.
        len: usize,
        /// Hard record ceiling.
        max: usize,
    },
    /// Scratch files would exceed the shared bounded disk budget.
    #[error("owner rewrite scratch budget exceeded: used={used} additional={additional} cap={cap}")]
    ScratchBudgetExceeded {
        /// Accounted scratch bytes.
        used: u64,
        /// New run bytes requested.
        additional: u64,
        /// Hard shared ceiling.
        cap: u64,
    },
}

/// Shared byte-accounting gate for every sorter in one migration.
#[derive(Debug)]
pub struct OwnerRewriteScratchBudget {
    cap: u64,
    used: AtomicU64,
    peak: AtomicU64,
}

impl OwnerRewriteScratchBudget {
    /// New shared budget.
    #[must_use]
    pub const fn new(cap: u64) -> Self {
        Self {
            cap,
            used: AtomicU64::new(0),
            peak: AtomicU64::new(0),
        }
    }

    /// Hard byte ceiling.
    #[must_use]
    pub const fn cap(&self) -> u64 {
        self.cap
    }

    /// Peak scratch bytes observed.
    #[must_use]
    pub fn peak(&self) -> u64 {
        self.peak.load(AtomicOrdering::Acquire)
    }

    fn reserve(&self, additional: u64) -> Result<(), OwnerRewriteError> {
        let mut used = self.used.load(AtomicOrdering::Acquire);
        loop {
            let next = used.saturating_add(additional);
            if next > self.cap {
                return Err(OwnerRewriteError::ScratchBudgetExceeded {
                    used,
                    additional,
                    cap: self.cap,
                });
            }
            match self.used.compare_exchange_weak(
                used,
                next,
                AtomicOrdering::AcqRel,
                AtomicOrdering::Acquire,
            ) {
                Ok(_) => {
                    self.peak.fetch_max(next, AtomicOrdering::AcqRel);
                    return Ok(());
                }
                Err(observed) => used = observed,
            }
        }
    }

    fn release(&self, bytes: u64) {
        self.used.fetch_sub(bytes, AtomicOrdering::AcqRel);
    }
}

#[derive(Debug, Clone)]
struct Run {
    path: PathBuf,
    bytes: u64,
    count: u64,
}

/// Lexicographic byte-record external sorter.
#[derive(Debug)]
pub struct BoundedOwnerSorter {
    dir: PathBuf,
    prefix: String,
    buffer: Vec<Vec<u8>>,
    buffer_bytes: usize,
    peak_buffer_bytes: usize,
    buffer_cap: usize,
    runs: Vec<Run>,
    next_run: u64,
    budget: Arc<OwnerRewriteScratchBudget>,
}

impl BoundedOwnerSorter {
    /// Create one class sorter under a migration-owned scratch directory.
    pub fn new(
        root: &Path,
        prefix: impl Into<String>,
        buffer_cap: usize,
        budget: Arc<OwnerRewriteScratchBudget>,
    ) -> Result<Self, OwnerRewriteError> {
        let prefix = prefix.into();
        let dir = root.join(&prefix);
        fs::create_dir_all(&dir)?;
        sync_dir(root)?;
        Ok(Self {
            dir,
            prefix,
            buffer: Vec::new(),
            buffer_bytes: 0,
            peak_buffer_bytes: 0,
            buffer_cap: buffer_cap.max(1),
            runs: Vec::new(),
            next_run: 0,
            budget,
        })
    }

    /// Push one record. The caller places its sort key first in the bytes.
    pub fn push(&mut self, record: Vec<u8>) -> Result<(), OwnerRewriteError> {
        if record.len() > OWNER_REWRITE_MAX_RECORD_BYTES {
            return Err(OwnerRewriteError::RecordTooLarge {
                len: record.len(),
                max: OWNER_REWRITE_MAX_RECORD_BYTES,
            });
        }
        let accounted = record.len().saturating_add(std::mem::size_of::<Vec<u8>>());
        if !self.buffer.is_empty() && self.buffer_bytes.saturating_add(accounted) > self.buffer_cap
        {
            self.flush_buffer()?;
        }
        self.buffer_bytes = self.buffer_bytes.saturating_add(accounted);
        self.peak_buffer_bytes = self.peak_buffer_bytes.max(self.buffer_bytes);
        self.buffer.push(record);
        Ok(())
    }

    /// Peak bytes retained in the in-memory sort chunk. This is independent
    /// of the total record count and exists so the rewrite gate can prove the
    /// implementation did not regress to collect-all-then-sort.
    #[must_use]
    pub const fn peak_resident_buffer_bytes(&self) -> usize {
        self.peak_buffer_bytes
    }

    /// Finish all runs, stream records in order through `visit`, and remove
    /// the migration scratch. Callback memory is one record.
    pub fn finish_visit(
        mut self,
        mut visit: impl FnMut(&[u8]) -> Result<(), OwnerRewriteError>,
    ) -> Result<(), OwnerRewriteError> {
        self.flush_buffer()?;
        while self.runs.len() > 1 {
            let old = std::mem::take(&mut self.runs);
            let mut next = Vec::new();
            for group in old.chunks(MERGE_FAN_IN) {
                if group.len() == 1 {
                    next.push(group[0].clone());
                } else {
                    next.push(self.merge_group(group)?);
                    for run in group {
                        self.remove_run(run)?;
                    }
                }
            }
            self.runs = next;
        }
        if let Some(run) = self.runs.pop() {
            let mut reader = RunReader::open(&run)?;
            while let Some(record) = reader.next_record()? {
                visit(&record)?;
            }
            reader.finish()?;
            self.remove_run(&run)?;
        }
        fs::remove_dir(&self.dir)?;
        if let Some(parent) = self.dir.parent() {
            sync_dir(parent)?;
        }
        Ok(())
    }

    fn flush_buffer(&mut self) -> Result<(), OwnerRewriteError> {
        if self.buffer.is_empty() {
            return Ok(());
        }
        self.buffer.sort_unstable();
        let records = std::mem::take(&mut self.buffer);
        self.buffer_bytes = 0;
        let estimated = encoded_run_bytes(records.iter().map(Vec::len))?;
        self.budget.reserve(estimated)?;
        let path = self.next_path();
        match write_run(
            &path,
            records.iter().map(Vec::as_slice),
            records.len() as u64,
        ) {
            Ok(()) => self.runs.push(Run {
                path,
                bytes: estimated,
                count: records.len() as u64,
            }),
            Err(error) => {
                self.budget.release(estimated);
                return Err(error);
            }
        }
        Ok(())
    }

    fn merge_group(&mut self, group: &[Run]) -> Result<Run, OwnerRewriteError> {
        let count: u64 = group.iter().map(|run| run.count).sum();
        let additional = group.iter().try_fold(RUN_HEADER_BYTES, |total, run| {
            total
                .checked_add(run.bytes.saturating_sub(RUN_HEADER_BYTES))
                .ok_or_else(|| OwnerRewriteError::Corrupt("merged run size wraps".to_owned()))
        })?;
        self.budget.reserve(additional)?;
        let path = self.next_path();
        let result = (|| {
            let mut readers: Vec<_> = group
                .iter()
                .map(RunReader::open)
                .collect::<Result<_, _>>()?;
            let mut heap = BinaryHeap::new();
            for (index, reader) in readers.iter_mut().enumerate() {
                if let Some(record) = reader.next_record()? {
                    heap.push(Reverse(HeapRecord { record, index }));
                }
            }
            let mut writer = RunWriter::create(&path, count)?;
            while let Some(Reverse(item)) = heap.pop() {
                writer.write_record(&item.record)?;
                if let Some(record) = readers[item.index].next_record()? {
                    heap.push(Reverse(HeapRecord {
                        record,
                        index: item.index,
                    }));
                }
            }
            for reader in readers {
                reader.finish()?;
            }
            writer.finish()?;
            Ok::<(), OwnerRewriteError>(())
        })();
        if let Err(error) = result {
            self.budget.release(additional);
            let _ = fs::remove_file(&path);
            return Err(error);
        }
        Ok(Run {
            path,
            bytes: additional,
            count,
        })
    }

    fn remove_run(&self, run: &Run) -> Result<(), OwnerRewriteError> {
        fs::remove_file(&run.path)?;
        self.budget.release(run.bytes);
        Ok(())
    }

    fn next_path(&mut self) -> PathBuf {
        let id = self.next_run;
        self.next_run = self.next_run.saturating_add(1);
        self.dir.join(format!("{}-{id:020}.run", self.prefix))
    }
}

impl Drop for BoundedOwnerSorter {
    fn drop(&mut self) {
        for run in &self.runs {
            if fs::remove_file(&run.path).is_ok() {
                self.budget.release(run.bytes);
            }
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
struct HeapRecord {
    record: Vec<u8>,
    index: usize,
}

impl Ord for HeapRecord {
    fn cmp(&self, other: &Self) -> Ordering {
        self.record
            .cmp(&other.record)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for HeapRecord {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct RunWriter {
    file: BufWriter<File>,
    expected_count: u64,
    written: u64,
    crc: u32,
}

impl RunWriter {
    fn create(path: &Path, count: u64) -> Result<Self, OwnerRewriteError> {
        let mut file = BufWriter::new(OpenOptions::new().write(true).create_new(true).open(path)?);
        file.write_all(RUN_MAGIC)?;
        file.write_all(&count.to_le_bytes())?;
        Ok(Self {
            file,
            expected_count: count,
            written: 0,
            crc: 0,
        })
    }

    fn write_record(&mut self, record: &[u8]) -> Result<(), OwnerRewriteError> {
        let len = u32::try_from(record.len()).map_err(|_| OwnerRewriteError::RecordTooLarge {
            len: record.len(),
            max: u32::MAX as usize,
        })?;
        let prefix = len.to_le_bytes();
        self.file.write_all(&prefix)?;
        self.file.write_all(record)?;
        self.crc = crc32c::crc32c_append(self.crc, &prefix);
        self.crc = crc32c::crc32c_append(self.crc, record);
        self.written += 1;
        Ok(())
    }

    fn finish(mut self) -> Result<(), OwnerRewriteError> {
        if self.written != self.expected_count {
            return Err(OwnerRewriteError::Corrupt(format!(
                "run writer expected {} records, wrote {}",
                self.expected_count, self.written
            )));
        }
        self.file.write_all(&self.crc.to_le_bytes())?;
        let raw = self.file.into_inner().map_err(|error| error.into_error())?;
        raw.sync_all()?;
        Ok(())
    }
}

struct RunReader {
    file: BufReader<File>,
    remaining: u64,
    crc: u32,
}

impl RunReader {
    fn open(run: &Run) -> Result<Self, OwnerRewriteError> {
        let mut file = BufReader::new(File::open(&run.path)?);
        let mut header = [0_u8; RUN_HEADER_BYTES as usize];
        file.read_exact(&mut header)?;
        if &header[..8] != RUN_MAGIC {
            return Err(OwnerRewriteError::Corrupt(format!(
                "bad run magic in {}",
                run.path.display()
            )));
        }
        let count = u64::from_le_bytes(
            header[8..16]
                .try_into()
                .map_err(|_| OwnerRewriteError::Corrupt("malformed run count".to_owned()))?,
        );
        if count != run.count {
            return Err(OwnerRewriteError::Corrupt(format!(
                "run {} count changed",
                run.path.display()
            )));
        }
        Ok(Self {
            file,
            remaining: count,
            crc: 0,
        })
    }

    fn next_record(&mut self) -> Result<Option<Vec<u8>>, OwnerRewriteError> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let mut prefix = [0_u8; 4];
        self.file.read_exact(&mut prefix)?;
        let len = u32::from_le_bytes(prefix) as usize;
        if len > OWNER_REWRITE_MAX_RECORD_BYTES {
            return Err(OwnerRewriteError::RecordTooLarge {
                len,
                max: OWNER_REWRITE_MAX_RECORD_BYTES,
            });
        }
        let mut record = vec![0_u8; len];
        self.file.read_exact(&mut record)?;
        self.crc = crc32c::crc32c_append(self.crc, &prefix);
        self.crc = crc32c::crc32c_append(self.crc, &record);
        self.remaining -= 1;
        Ok(Some(record))
    }

    fn finish(mut self) -> Result<(), OwnerRewriteError> {
        if self.remaining != 0 {
            return Err(OwnerRewriteError::Corrupt(
                "run reader finished before all records".to_owned(),
            ));
        }
        let mut footer = [0_u8; 4];
        self.file.read_exact(&mut footer)?;
        if u32::from_le_bytes(footer) != self.crc {
            return Err(OwnerRewriteError::Corrupt(
                "run checksum mismatch".to_owned(),
            ));
        }
        let mut trailing = [0_u8; 1];
        if self.file.read(&mut trailing)? != 0 {
            return Err(OwnerRewriteError::Corrupt(
                "run carries trailing bytes".to_owned(),
            ));
        }
        Ok(())
    }
}

fn write_run<'a>(
    path: &Path,
    records: impl Iterator<Item = &'a [u8]>,
    count: u64,
) -> Result<(), OwnerRewriteError> {
    let mut writer = RunWriter::create(path, count)?;
    for record in records {
        writer.write_record(record)?;
    }
    writer.finish()
}

fn encoded_run_bytes(mut lengths: impl Iterator<Item = usize>) -> Result<u64, OwnerRewriteError> {
    lengths.try_fold(RUN_HEADER_BYTES + 4, |total, len| {
        let len = u64::try_from(len)
            .map_err(|_| OwnerRewriteError::Corrupt("record length overflows u64".to_owned()))?;
        total
            .checked_add(RECORD_PREFIX_BYTES)
            .and_then(|value| value.checked_add(len))
            .ok_or_else(|| OwnerRewriteError::Corrupt("run byte length wraps".to_owned()))
    })
}

fn sync_dir(path: &Path) -> Result<(), std::io::Error> {
    File::open(path)?.sync_all()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounded_runs_merge_in_order_and_cleanup() {
        let root = tempfile::tempdir().unwrap();
        let budget = Arc::new(OwnerRewriteScratchBudget::new(1024 * 1024));
        let mut sorter =
            BoundedOwnerSorter::new(root.path(), "bindings", 64, budget.clone()).unwrap();
        for value in ["z", "b", "a", "m", "a", "q", "c"] {
            sorter.push(value.as_bytes().to_vec()).unwrap();
        }
        let mut observed = Vec::new();
        sorter
            .finish_visit(|record| {
                observed.push(String::from_utf8(record.to_vec()).unwrap());
                Ok(())
            })
            .unwrap();
        assert_eq!(observed, ["a", "a", "b", "c", "m", "q", "z"]);
        assert!(budget.peak() <= budget.cap());
        assert!(root.path().read_dir().unwrap().next().is_none());
    }

    #[test]
    fn scratch_budget_bites_before_write() {
        let root = tempfile::tempdir().unwrap();
        let budget = Arc::new(OwnerRewriteScratchBudget::new(24));
        let mut sorter = BoundedOwnerSorter::new(root.path(), "tiny", 1, budget).unwrap();
        let error = sorter
            .push(vec![1; 16])
            .and_then(|_| sorter.finish_visit(|_| Ok(())))
            .unwrap_err();
        assert!(matches!(
            error,
            OwnerRewriteError::ScratchBudgetExceeded { .. }
        ));
    }

    #[test]
    fn rewriter_rss_bounded_sorted_run_merge() {
        const RECORDS: u64 = 20_000;
        const BUFFER_CAP: usize = 16 * 1024;
        const DISK_CAP: u64 = 8 * 1024 * 1024;

        let root = tempfile::tempdir().unwrap();
        let budget = Arc::new(OwnerRewriteScratchBudget::new(DISK_CAP));
        let mut sorter =
            BoundedOwnerSorter::new(root.path(), "rss-gate", BUFFER_CAP, budget.clone()).unwrap();
        for value in (0..RECORDS).rev() {
            sorter.push(value.to_be_bytes().to_vec()).unwrap();
        }

        let one_record_accounting = 8 + std::mem::size_of::<Vec<u8>>();
        assert!(
            sorter.peak_resident_buffer_bytes() <= BUFFER_CAP + one_record_accounting,
            "resident sort chunk must remain independent of N"
        );

        let mut expected = 0_u64;
        sorter
            .finish_visit(|record| {
                let value = u64::from_be_bytes(record.try_into().map_err(|_| {
                    OwnerRewriteError::Corrupt("rewriter gate record width changed".to_owned())
                })?);
                assert_eq!(value, expected);
                expected += 1;
                Ok(())
            })
            .unwrap();
        assert_eq!(expected, RECORDS);
        assert!(budget.peak() <= DISK_CAP, "8 MiB scratch cap must bite");
        assert!(root.path().read_dir().unwrap().next().is_none());
    }
}
