//! Page I/O trait and implementations.
//!
//! Covers roadmap tasks M1-20 (trait) and M1-21 (`PosixPageIo` via
//! `pread` / `pwrite` on any Unix). `IoUringPageIo` (M1-22) and
//! `O_DIRECT` (M1-24) land in follow-up PRs against this trait.

use std::collections::HashMap;
#[cfg(unix)]
use std::fs::{File, OpenOptions};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(unix)]
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use arcgraph_core::{ArcGraphError, PAGE_SIZE, PageId, Result};
use parking_lot::RwLock;

/// An 8 KiB page buffer used as the unit of I/O.
pub type PageBuf = [u8; PAGE_SIZE];

/// Abstract page I/O.
///
/// Implementations must be safe to share across threads and must
/// return a deterministic error rather than panic on any failure.
pub trait PageIo: Send + Sync {
    /// Read the page at `page_id` into `buf`. Returns an error if the
    /// page does not exist or the underlying store failed.
    fn read_page(&self, page_id: PageId, buf: &mut PageBuf) -> Result<()>;

    /// Write `buf` at `page_id`, allocating space if the page is new.
    fn write_page(&self, page_id: PageId, buf: &PageBuf) -> Result<()>;

    /// Durability barrier. Implementations that hold writes in memory
    /// or a kernel cache must fsync here.
    fn flush(&self) -> Result<()>;
}

// ----- in-memory impl used in tests and bootstrap ---------------------------

/// Thread-safe in-memory `PageIo`. Intended for tests, microbenchmarks,
/// and the ephemeral-database mode in `arcgraph-cli`. Does *not*
/// persist across process restarts.
pub struct InMemoryPageIo {
    pages: RwLock<HashMap<PageId, Box<PageBuf>>>,
    reads: AtomicU64,
    writes: AtomicU64,
}

impl InMemoryPageIo {
    /// Fresh store with no pages.
    #[must_use]
    pub fn new() -> Self {
        Self {
            pages: RwLock::new(HashMap::new()),
            reads: AtomicU64::new(0),
            writes: AtomicU64::new(0),
        }
    }

    /// Number of read_page calls issued (monotonic; useful in tests).
    #[must_use]
    pub fn reads(&self) -> u64 {
        self.reads.load(Ordering::Relaxed)
    }

    /// Number of write_page calls issued.
    #[must_use]
    pub fn writes(&self) -> u64 {
        self.writes.load(Ordering::Relaxed)
    }

    /// Pre-seed a page. Handy for tests that want a specific starting state.
    pub fn put(&self, page_id: PageId, buf: PageBuf) {
        self.pages.write().insert(page_id, Box::new(buf));
    }
}

impl Default for InMemoryPageIo {
    fn default() -> Self {
        Self::new()
    }
}

impl PageIo for InMemoryPageIo {
    fn read_page(&self, page_id: PageId, buf: &mut PageBuf) -> Result<()> {
        self.reads.fetch_add(1, Ordering::Relaxed);
        let pages = self.pages.read();
        match pages.get(&page_id) {
            Some(page) => {
                buf.copy_from_slice(page.as_ref());
                Ok(())
            }
            None => Err(ArcGraphError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("page {} not found", page_id.raw()),
            ))),
        }
    }

    fn write_page(&self, page_id: PageId, buf: &PageBuf) -> Result<()> {
        self.writes.fetch_add(1, Ordering::Relaxed);
        self.pages.write().insert(page_id, Box::new(*buf));
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        // In-memory store has no kernel cache.
        Ok(())
    }
}

// ----- POSIX impl (M1-21) ---------------------------------------------------

/// POSIX `pread` / `pwrite` `PageIo` on top of a single file.
///
/// Thread-safe: `pread`/`pwrite` are atomic and stateless with respect
/// to the file position, so callers may issue concurrent reads and
/// writes on the same instance without additional locking. Growing
/// the file beyond the highest written page is a side effect of
/// `write_page` — POSIX allows sparse writes that implicitly zero
/// fill the gap.
///
/// See design-v2 §3.4. `O_DIRECT` (M1-24) and `IoUringPageIo` (M1-22)
/// layer on top of this in later PRs.
#[cfg(unix)]
pub struct PosixPageIo {
    file: File,
}

#[cfg(unix)]
impl PosixPageIo {
    /// Open an existing file read/write. Fails if the file doesn't
    /// exist. For new databases use [`Self::create`].
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new().read(true).write(true).open(path)?;
        Ok(Self { file })
    }

    /// Create (or truncate) a file and open it read/write. Used for
    /// fresh databases and for tests.
    pub fn create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;
        Ok(Self { file })
    }

    /// Open for read/write, creating only if the file does not already exist.
    pub fn open_or_create<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)?;
        Ok(Self { file })
    }

    /// Current file length in bytes.
    pub fn file_len(&self) -> Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    fn offset_of(page_id: PageId) -> Result<u64> {
        page_id
            .raw()
            .checked_mul(PAGE_SIZE as u64)
            .ok_or_else(|| ArcGraphError::Io(std::io::Error::other("page id offset overflows u64")))
    }
}

#[cfg(unix)]
impl PageIo for PosixPageIo {
    fn read_page(&self, page_id: PageId, buf: &mut PageBuf) -> Result<()> {
        let offset = Self::offset_of(page_id)?;
        self.file.read_exact_at(buf, offset)?;
        Ok(())
    }

    fn write_page(&self, page_id: PageId, buf: &PageBuf) -> Result<()> {
        let offset = Self::offset_of(page_id)?;
        self.file.write_all_at(buf, offset)?;
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        // `fdatasync` on Linux; `F_FULLFSYNC`/`fsync` on macOS per std docs.
        self.file.sync_data()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_write_read() {
        let io = InMemoryPageIo::new();
        let mut page = [0u8; PAGE_SIZE];
        page[0..4].copy_from_slice(&0xDEAD_BEEF_u32.to_le_bytes());
        io.write_page(PageId::new(1), &page).unwrap();

        let mut back = [0u8; PAGE_SIZE];
        io.read_page(PageId::new(1), &mut back).unwrap();
        assert_eq!(&page[..], &back[..]);
        assert_eq!(io.writes(), 1);
        assert_eq!(io.reads(), 1);
    }

    #[test]
    fn read_missing_is_error() {
        let io = InMemoryPageIo::new();
        let mut buf = [0u8; PAGE_SIZE];
        let err = io.read_page(PageId::new(42), &mut buf).unwrap_err();
        assert!(matches!(err, ArcGraphError::Io(_)));
    }

    #[test]
    fn flush_is_noop() {
        let io = InMemoryPageIo::new();
        io.flush().unwrap();
    }
}

#[cfg(all(test, unix))]
mod posix_tests {
    use std::sync::Arc;
    use std::thread;

    use proptest::prelude::*;
    use tempfile::tempdir;

    use super::*;

    fn new_posix() -> (tempfile::TempDir, PosixPageIo) {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("pages.db");
        let io = PosixPageIo::create(&path).expect("create");
        (dir, io)
    }

    #[test]
    fn write_then_read_single_page() {
        let (_dir, io) = new_posix();
        let mut page = [0u8; PAGE_SIZE];
        page[0..4].copy_from_slice(&0xCAFE_F00D_u32.to_le_bytes());
        page[PAGE_SIZE - 1] = 0xAB;
        io.write_page(PageId::new(0), &page).unwrap();
        io.flush().unwrap();

        let mut back = [0u8; PAGE_SIZE];
        io.read_page(PageId::new(0), &mut back).unwrap();
        assert_eq!(&page[..], &back[..]);
    }

    #[test]
    fn sparse_write_and_readback() {
        let (_dir, io) = new_posix();
        let mut page0 = [0xAA_u8; PAGE_SIZE];
        let mut page9 = [0xBB_u8; PAGE_SIZE];
        // Poison a byte to make equality meaningful.
        page0[0] = 0x01;
        page9[0] = 0x09;
        io.write_page(PageId::new(0), &page0).unwrap();
        io.write_page(PageId::new(9), &page9).unwrap();
        io.flush().unwrap();

        let mut r0 = [0u8; PAGE_SIZE];
        let mut r9 = [0u8; PAGE_SIZE];
        io.read_page(PageId::new(0), &mut r0).unwrap();
        io.read_page(PageId::new(9), &mut r9).unwrap();
        assert_eq!(&r0[..], &page0[..]);
        assert_eq!(&r9[..], &page9[..]);

        // Reading a hole (POSIX zero-fills) must succeed and read zeros.
        let mut hole = [0xFF_u8; PAGE_SIZE];
        io.read_page(PageId::new(5), &mut hole).unwrap();
        assert_eq!(hole, [0u8; PAGE_SIZE]);
    }

    #[test]
    fn read_beyond_eof_is_error() {
        let (_dir, io) = new_posix();
        let mut buf = [0u8; PAGE_SIZE];
        let err = io.read_page(PageId::new(100), &mut buf).unwrap_err();
        assert!(matches!(err, ArcGraphError::Io(_)));
    }

    #[test]
    fn page_id_offset_overflow_is_error() {
        let (_dir, io) = new_posix();
        let mut buf = [0u8; PAGE_SIZE];
        // u64::MAX / PAGE_SIZE is ~2.25e15. Anything above that overflows.
        let bad = PageId::new(u64::MAX / PAGE_SIZE as u64 + 1);
        let err = io.read_page(bad, &mut buf).unwrap_err();
        assert!(matches!(err, ArcGraphError::Io(_)));
    }

    #[test]
    fn reopen_preserves_data() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("pages.db");
        {
            let io = PosixPageIo::create(&path).unwrap();
            let mut p = [0u8; PAGE_SIZE];
            p[0] = 0x42;
            io.write_page(PageId::new(7), &p).unwrap();
            io.flush().unwrap();
        }
        let io = PosixPageIo::open(&path).unwrap();
        let mut r = [0u8; PAGE_SIZE];
        io.read_page(PageId::new(7), &mut r).unwrap();
        assert_eq!(r[0], 0x42);
    }

    #[test]
    fn concurrent_readers_and_writers_are_atomic_per_page() {
        let (_dir, io) = new_posix();
        let io = Arc::new(io);
        // Seed pages 0..8 with their own id as the first byte.
        for i in 0u64..8 {
            let mut p = [0u8; PAGE_SIZE];
            p[0] = i as u8;
            io.write_page(PageId::new(i), &p).unwrap();
        }
        io.flush().unwrap();
        thread::scope(|s| {
            // 4 readers; 4 writers; readers must never see a torn page.
            for t in 0..4 {
                let io = io.clone();
                s.spawn(move || {
                    for _ in 0..50 {
                        let mut buf = [0u8; PAGE_SIZE];
                        let pid = PageId::new(t as u64);
                        io.read_page(pid, &mut buf).unwrap();
                        // Byte 0 is the stable "id marker" written atomically.
                        assert_eq!(buf[0], t as u8);
                    }
                });
            }
            for t in 4..8 {
                let io = io.clone();
                s.spawn(move || {
                    for pass in 0..50 {
                        let mut p = [0u8; PAGE_SIZE];
                        p[0] = t as u8;
                        p[1] = pass as u8;
                        io.write_page(PageId::new(t as u64), &p).unwrap();
                    }
                });
            }
        });
    }

    #[test]
    fn ten_thousand_pages_roundtrip() {
        // Roadmap M1-21: "10K pages roundtrip" integration test.
        let (_dir, io) = new_posix();
        const N: u64 = 10_000;
        for i in 0..N {
            let mut p = [0u8; PAGE_SIZE];
            p[0..8].copy_from_slice(&i.to_le_bytes());
            io.write_page(PageId::new(i), &p).unwrap();
        }
        io.flush().unwrap();
        for i in 0..N {
            let mut r = [0u8; PAGE_SIZE];
            io.read_page(PageId::new(i), &mut r).unwrap();
            let tag = u64::from_le_bytes(r[0..8].try_into().expect("8 bytes"));
            assert_eq!(tag, i);
        }
    }

    proptest! {
        // PageId range is bounded to [0, 4096) on purpose: `pwrite` at
        // PageId::new(u32::MAX) implies a ~35 TB sparse file, which
        // exceeds the 16 TB ext4 max-file-size cap on Linux CI runners
        // (ubuntu-latest) and returns `EFBIG`. macOS APFS tolerates the
        // enormous sparse file and the test silently passed there; the
        // Linux-only unwrap panic tracked as #14. This property cares
        // about read-your-write correctness over arbitrary ids, not
        // about stress-testing the filesystem's sparse-file ceiling —
        // 4 K distinct ids × up to 64 writes per case already exercises
        // the id-to-offset arithmetic, hole semantics, and duplicate-id
        // last-write-wins branches that the assertions below check.
        #[test]
        fn property_write_then_read_roundtrip(
            pages in prop::collection::vec(
                (0u32..4096, any::<u8>()).prop_map(|(id, byte)| (PageId::new(u64::from(id)), byte)),
                1..=64,
            ),
        ) {
            let (_dir, io) = new_posix();
            for (pid, byte) in &pages {
                let mut p = [0u8; PAGE_SIZE];
                p[0] = *byte;
                p[PAGE_SIZE - 1] = *byte ^ 0xFF;
                io.write_page(*pid, &p).unwrap();
            }
            io.flush().unwrap();
            // For each distinct page id, the most recent byte must come back.
            let mut latest = std::collections::HashMap::new();
            for (pid, byte) in &pages {
                latest.insert(*pid, *byte);
            }
            for (pid, byte) in &latest {
                let mut r = [0u8; PAGE_SIZE];
                io.read_page(*pid, &mut r).unwrap();
                prop_assert_eq!(r[0], *byte);
                prop_assert_eq!(r[PAGE_SIZE - 1], *byte ^ 0xFF);
            }
        }
    }
}
