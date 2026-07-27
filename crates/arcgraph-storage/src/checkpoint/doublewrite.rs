//! Checksummed fixed-batch doublewrite area for M3 home-page flushes.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use arcgraph_core::record::{PAGE_SIZE, PageHeader};
use arcgraph_core::{ArcGraphError, Lsn, Result, TenantId};

use crate::io::PageBuf;
use crate::io::PageIo;
use crate::page_store::TenantPageIo;
use crate::wal::{STORE_BLOB_OVERFLOW, STORE_PROPS, STORE_RECORD, STORE_TEL};

pub const DOUBLEWRITE_FILE: &str = "pages.doublewrite";
const MAGIC: [u8; 8] = *b"AGDWB001";
const VERSION: u16 = 1;
const HEADER_LEN: usize = 32;
const ENTRY_HEADER_LEN: usize = 32;
const ENTRY_LEN: usize = ENTRY_HEADER_LEN + PAGE_SIZE;
type DoublewritePage = (DoublewriteKey, Box<PageBuf>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DoublewriteKey {
    pub tenant_id: TenantId,
    pub store_id: u16,
    pub page_no: u64,
}

pub trait DoublewriteRestoreTarget {
    fn read_home(&mut self, key: DoublewriteKey) -> Result<Option<Box<PageBuf>>>;
    fn write_home(&mut self, key: DoublewriteKey, page: &PageBuf) -> Result<()>;
    fn sync_home(&mut self) -> Result<()>;
}

/// Production M3 restore target over the two physical home files.
pub struct M3DoublewriteHome {
    props: std::sync::Arc<dyn PageIo>,
    records: RecordsHome,
    touched_record_homes: std::collections::BTreeMap<TenantId, std::sync::Arc<dyn PageIo>>,
    extent_directories: BTreeMap<(TenantId, u16), std::sync::Arc<crate::extent::ExtentDirectory>>,
}

enum RecordsHome {
    Shared(std::sync::Arc<dyn PageIo>),
    PerTenant(std::sync::Arc<dyn TenantPageIo>),
}

impl M3DoublewriteHome {
    #[must_use]
    pub fn new(props: std::sync::Arc<dyn PageIo>, records: std::sync::Arc<dyn PageIo>) -> Self {
        Self {
            props,
            records: RecordsHome::Shared(records),
            touched_record_homes: std::collections::BTreeMap::new(),
            extent_directories: BTreeMap::new(),
        }
    }

    #[must_use]
    pub fn with_tenant_records(
        props: std::sync::Arc<dyn PageIo>,
        records: std::sync::Arc<dyn TenantPageIo>,
    ) -> Self {
        Self {
            props,
            records: RecordsHome::PerTenant(records),
            touched_record_homes: std::collections::BTreeMap::new(),
            extent_directories: BTreeMap::new(),
        }
    }

    /// Register a production extent directory so tagged DWB keys restore
    /// through their fixed head offsets during the ordinary CLI bootstrap.
    #[must_use]
    pub fn with_extent_directory(
        mut self,
        directory: std::sync::Arc<crate::extent::ExtentDirectory>,
    ) -> Self {
        self.extent_directories
            .insert((directory.tenant(), directory.store_id()), directory);
        self
    }

    fn extent_directory(
        &self,
        key: DoublewriteKey,
    ) -> Result<&std::sync::Arc<crate::extent::ExtentDirectory>> {
        self.extent_directories
            .get(&(key.tenant_id, key.store_id))
            .ok_or_else(|| corruption("tagged DWB key has no production extent directory"))
    }

    fn mapped_extent_directory(
        &self,
        key: DoublewriteKey,
    ) -> Result<Option<&std::sync::Arc<crate::extent::ExtentDirectory>>> {
        let Some(directory) = self.extent_directories.get(&(key.tenant_id, key.store_id)) else {
            return Ok(None);
        };
        directory
            .mapping(key.page_no / crate::extent::EXTENT_PAGES)
            .map(|mapping| mapping.map(|_| directory))
    }

    fn store(&self, key: DoublewriteKey) -> Result<std::sync::Arc<dyn PageIo>> {
        match key.store_id {
            STORE_PROPS => Ok(std::sync::Arc::clone(&self.props)),
            STORE_RECORD => match &self.records {
                RecordsHome::Shared(io) => Ok(std::sync::Arc::clone(io)),
                RecordsHome::PerTenant(io) => io.io_for(key.tenant_id),
            },
            other => Err(corruption(format!(
                "DWB restore target received unsupported store {other}"
            ))),
        }
    }
}

impl DoublewriteRestoreTarget for M3DoublewriteHome {
    fn read_home(&mut self, key: DoublewriteKey) -> Result<Option<Box<PageBuf>>> {
        if crate::extent::is_directory_page(key.page_no) {
            return self.extent_directory(key)?.read_home_page(key.page_no);
        }
        if let Some(directory) = self.mapped_extent_directory(key)? {
            return directory.read_data_home_page(key.page_no);
        }
        let mut page = Box::new([0u8; PAGE_SIZE]);
        match self
            .store(key)?
            .read_page(arcgraph_core::PageId::new(key.page_no), page.as_mut())
        {
            Ok(()) => Ok(Some(page)),
            Err(ArcGraphError::Io(error)) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn write_home(&mut self, key: DoublewriteKey, page: &PageBuf) -> Result<()> {
        if crate::extent::is_directory_page(key.page_no) {
            return self
                .extent_directory(key)?
                .write_home_page(key.page_no, page);
        }
        if let Some(directory) = self.mapped_extent_directory(key)? {
            return directory.write_data_home_page(key.page_no, page);
        }
        let io = self.store(key)?;
        io.write_page(arcgraph_core::PageId::new(key.page_no), page)?;
        if key.store_id == STORE_RECORD {
            self.touched_record_homes.insert(key.tenant_id, io);
        }
        Ok(())
    }

    fn sync_home(&mut self) -> Result<()> {
        self.props.flush()?;
        for io in self.touched_record_homes.values() {
            io.flush()?;
        }
        for directory in self.extent_directories.values() {
            directory.sync_home()?;
        }
        Ok(())
    }
}

/// Doublewrite restore target for tagged extent-directory pages.
///
/// The registry cardinality is `(tenant, store)` owners, not extents. Every
/// read/write delegates to [`crate::extent::ExtentDirectory`], preserving the
/// fixed-head bootstrap map instead of treating a tagged page as an ordinary
/// physical file page.
#[derive(Default)]
pub struct ExtentDirectoryDoublewriteHome {
    directories: BTreeMap<(TenantId, u16), std::sync::Arc<crate::extent::ExtentDirectory>>,
}

impl ExtentDirectoryDoublewriteHome {
    /// Construct an empty directory restore registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register one tenant/store directory owner.
    #[must_use]
    pub fn with_directory(
        mut self,
        directory: std::sync::Arc<crate::extent::ExtentDirectory>,
    ) -> Self {
        self.directories
            .insert((directory.tenant(), directory.store_id()), directory);
        self
    }

    fn directory(
        &self,
        key: DoublewriteKey,
    ) -> Result<&std::sync::Arc<crate::extent::ExtentDirectory>> {
        if !crate::extent::is_directory_page(key.page_no) {
            return Err(corruption(
                "extent-directory DWB target received an untagged data page",
            ));
        }
        self.directories
            .get(&(key.tenant_id, key.store_id))
            .ok_or_else(|| corruption("extent-directory DWB target has no matching owner"))
    }
}

impl DoublewriteRestoreTarget for ExtentDirectoryDoublewriteHome {
    fn read_home(&mut self, key: DoublewriteKey) -> Result<Option<Box<PageBuf>>> {
        self.directory(key)?.read_home_page(key.page_no)
    }

    fn write_home(&mut self, key: DoublewriteKey, page: &PageBuf) -> Result<()> {
        self.directory(key)?.write_home_page(key.page_no, page)
    }

    fn sync_home(&mut self) -> Result<()> {
        for directory in self.directories.values() {
            directory.sync_home()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DoublewriteRestoreReport {
    pub valid_slots: usize,
    pub restored_pages: usize,
    pub skipped_newer_homes: usize,
    pub ignored_torn_batch: bool,
}

#[derive(Debug)]
pub struct DoublewriteArea {
    path: PathBuf,
    serial: Mutex<()>,
}

impl DoublewriteArea {
    #[must_use]
    pub fn new(data_dir: &Path) -> Self {
        Self {
            path: data_dir.join(DOUBLEWRITE_FILE),
            serial: Mutex::new(()),
        }
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Keys in the last complete, checksummed batch. A missing or torn batch
    /// has no eligible pages and therefore returns an empty census.
    pub fn valid_batch_keys(&self) -> Result<Vec<DoublewriteKey>> {
        let mut keys: Vec<_> = self
            .read_valid_batch()?
            .unwrap_or_default()
            .into_iter()
            .map(|(key, _)| key)
            .collect();
        keys.sort_unstable();
        Ok(keys)
    }

    /// Replace the fixed DWB batch and fsync it before returning. Callers may
    /// begin home writes only after this succeeds.
    pub fn stage_batch(&self, pages: &[(DoublewriteKey, &PageBuf)]) -> Result<()> {
        let _serial = self.serial.lock().expect("doublewrite mutex poisoned");
        let count =
            u32::try_from(pages.len()).map_err(|_| corruption("DWB page count overflow"))?;
        let payload_len = pages
            .len()
            .checked_mul(ENTRY_LEN)
            .ok_or_else(|| corruption("DWB batch length overflow"))?;
        let mut payload = Vec::with_capacity(payload_len);
        for (key, page) in pages {
            validate_page(*key, page)?;
            payload.extend_from_slice(&key.store_id.to_le_bytes());
            payload.extend_from_slice(&0u16.to_le_bytes());
            payload.extend_from_slice(&key.tenant_id.raw().to_le_bytes());
            payload.extend_from_slice(&key.page_no.to_le_bytes());
            let page_lsn = u64::from_le_bytes(page[16..24].try_into().unwrap());
            payload.extend_from_slice(&page_lsn.to_le_bytes());
            payload.extend_from_slice(&crc32c::crc32c(*page).to_le_bytes());
            payload.extend_from_slice(*page);
        }

        let mut header = [0u8; HEADER_LEN];
        header[0..8].copy_from_slice(&MAGIC);
        header[8..10].copy_from_slice(&VERSION.to_le_bytes());
        header[12..16].copy_from_slice(&count.to_le_bytes());
        header[16..24].copy_from_slice(&(payload_len as u64).to_le_bytes());
        header[24..28].copy_from_slice(&crc32c::crc32c(&payload).to_le_bytes());
        let header_crc = crc32c::crc32c(&header[..28]);
        header[28..32].copy_from_slice(&header_crc.to_le_bytes());

        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&self.path)?;
        file.write_all(&header)?;
        file.write_all(&payload)?;
        file.sync_data()?;
        if let Some(parent) = self.path.parent() {
            File::open(parent)?.sync_all()?;
        }
        Ok(())
    }

    /// Restore torn or older home pages from the last complete DWB batch.
    /// A torn DWB batch is ignored as a unit: home writes cannot have begun
    /// before the corresponding DWB fsync completed.
    pub fn restore(
        &self,
        target: &mut dyn DoublewriteRestoreTarget,
    ) -> Result<DoublewriteRestoreReport> {
        let _serial = self.serial.lock().expect("doublewrite mutex poisoned");
        let Some(entries) = self.read_valid_batch()? else {
            return Ok(DoublewriteRestoreReport {
                ignored_torn_batch: self.path.exists(),
                ..DoublewriteRestoreReport::default()
            });
        };
        let mut report = DoublewriteRestoreReport {
            valid_slots: entries.len(),
            ..DoublewriteRestoreReport::default()
        };
        for (key, dwb_page) in entries {
            let dwb_lsn = page_lsn(&dwb_page);
            let restore = match target.read_home(key)? {
                None => true,
                Some(home) => match validate_page(key, &home) {
                    Ok(()) => page_lsn(&home).raw() < dwb_lsn.raw(),
                    Err(_) => true,
                },
            };
            if restore {
                target.write_home(key, &dwb_page)?;
                report.restored_pages += 1;
            } else {
                report.skipped_newer_homes += 1;
            }
        }
        if report.restored_pages > 0 {
            target.sync_home()?;
        }
        Ok(report)
    }

    fn read_valid_batch(&self) -> Result<Option<Vec<DoublewritePage>>> {
        let mut file = match File::open(&self.path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        if bytes.len() < HEADER_LEN || bytes[0..8] != MAGIC {
            return Ok(None);
        }
        let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
        if version != VERSION || bytes[10..12] != [0, 0] {
            return Ok(None);
        }
        let stored_header_crc = u32::from_le_bytes(bytes[28..32].try_into().unwrap());
        if crc32c::crc32c(&bytes[..28]) != stored_header_crc {
            return Ok(None);
        }
        let count = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let payload_len = u64::from_le_bytes(bytes[16..24].try_into().unwrap()) as usize;
        if payload_len != count.saturating_mul(ENTRY_LEN)
            || bytes.len() != HEADER_LEN.saturating_add(payload_len)
        {
            return Ok(None);
        }
        let payload = &bytes[HEADER_LEN..];
        let stored_payload_crc = u32::from_le_bytes(bytes[24..28].try_into().unwrap());
        if crc32c::crc32c(payload) != stored_payload_crc {
            return Ok(None);
        }
        let mut entries = Vec::with_capacity(count);
        for entry in payload.chunks_exact(ENTRY_LEN) {
            let store_id = u16::from_le_bytes(entry[0..2].try_into().unwrap());
            if entry[2..4] != [0, 0]
                || !matches!(
                    store_id,
                    STORE_PROPS | STORE_RECORD | STORE_TEL | STORE_BLOB_OVERFLOW
                )
            {
                return Ok(None);
            }
            let key = DoublewriteKey {
                store_id,
                tenant_id: TenantId::new(u64::from_le_bytes(entry[4..12].try_into().unwrap())),
                page_no: u64::from_le_bytes(entry[12..20].try_into().unwrap()),
            };
            let recorded_lsn = u64::from_le_bytes(entry[20..28].try_into().unwrap());
            let recorded_crc = u32::from_le_bytes(entry[28..32].try_into().unwrap());
            let mut page: Box<PageBuf> = Box::new([0; PAGE_SIZE]);
            page.copy_from_slice(&entry[ENTRY_HEADER_LEN..]);
            if crc32c::crc32c(page.as_ref()) != recorded_crc
                || page_lsn(&page).raw() != recorded_lsn
                || validate_page(key, &page).is_err()
            {
                return Ok(None);
            }
            entries.push((key, page));
        }
        Ok(Some(entries))
    }
}

fn validate_page(key: DoublewriteKey, page: &PageBuf) -> Result<()> {
    let header_bytes: &[u8; PageHeader::SIZE] = page[..PageHeader::SIZE].try_into().unwrap();
    let header = PageHeader::from_bytes(header_bytes)?;
    if header.page_id != key.page_no || header.tenant_id != key.tenant_id.raw() {
        return Err(corruption("DWB key does not match page header"));
    }
    if crc32c::crc32c(&page[PageHeader::SIZE..]) != header.checksum {
        return Err(corruption("DWB page body checksum mismatch"));
    }
    Ok(())
}

fn page_lsn(page: &PageBuf) -> Lsn {
    Lsn::new(u64::from_le_bytes(page[16..24].try_into().unwrap()))
}

fn corruption(reason: impl Into<String>) -> ArcGraphError {
    ArcGraphError::WalCorruption {
        lsn: Lsn::ZERO,
        reason: reason.into(),
    }
}
