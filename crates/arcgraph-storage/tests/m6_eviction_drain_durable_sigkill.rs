//! M6.1 — `m6_eviction_drain_durable_sigkill`.
//!
//! Real SIGKILL mid-eviction-drain → recovery lossless.
//!
//! A child subprocess opens a real disk-backed `BufferedRecordPageStore`
//! plus `DirtyPageTable` plus `WriteBehindCheckpointer`, under a TINY
//! cache cap (continuous `evict_for_capacity` pressure — MECH-E1..E8
//! firing on essentially every mutation), commits a sequence of page
//! mutations (mutate -> real WAL fsync -> DPT mark_dirty, mirroring the
//! production `crud.rs::apply_txn_slotted_deltas` shape used by the
//! other M6.1 gates) each recorded to a ledger file the moment its WAL
//! record is durable, then loops sleeping between commits so the
//! parent's crash window reliably lands mid-drain. The parent SIGKILLs
//! the child at a randomized point, then recovers: opens a FRESH store
//! bound to the SAME on-disk home file (no in-process state survives
//! SIGKILL) and confirms every ledgered (page_no, byte) pair reads back
//! correctly — i.e., eviction never dropped a page's durable image,
//! even mid-drain.
//!
//! Per `feedback_test_env_gate_panic_by_default.md`: this real-fault
//! test panics unless explicitly opted in or explicitly opted out —
//! never a silent soft-skip.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Once};
use std::time::Duration;

use arcgraph_core::{Lsn, PAGE_SIZE, PageId, PageType, TenantId};
use arcgraph_storage::checkpoint::{PageFlushTarget, WriteBehindCheckpointer};
use arcgraph_storage::io::{PageIo, PosixPageIo};
use arcgraph_storage::page_store::{
    BufferedRecordPageStore, PerTenantBufferPool, PerTenantBufferPoolConfig, RecordPageBackend,
};
use arcgraph_storage::redo::{DirtyPageKey, DirtyPageTable};
use arcgraph_storage::test_harness::k1::subprocess::{
    SubprocessWorkloadRegistry, WORKLOAD_CLEAN_EXIT_CODE, maybe_dispatch_subprocess_workload,
    run_with_crash_window_via_dispatcher,
};
use arcgraph_storage::wal::{STORE_RECORD, WalConfig, WalRecordType, WalWriter};

const WORKLOAD_NAME: &str = "m6_eviction_drain_durable_sigkill_workload";
const DISPATCHER_TEST: &str = "aaaa_subprocess_dispatcher_router";
const CACHE_CAP: usize = 8;
const TOTAL_PAGES: u64 = 400;
const SLEEP_BETWEEN_COMMITS: Duration = Duration::from_millis(5);
// Total commit budget ≈ TOTAL_PAGES * SLEEP_BETWEEN_COMMITS ≈ 2s —
// comfortably longer than the crash window so SIGKILL always fires
// mid-drain, never after clean completion.
const CRASH_WINDOW: Duration = Duration::from_millis(500);

const SKIP_ENV: &str = "ARCGRAPH_M6_EVICT_SIGKILL_SKIP_OK";
const RUN_ENV: &str = "ARCGRAPH_M6_EVICT_SIGKILL_RUN";

fn ledger_path(dir: &Path) -> PathBuf {
    dir.join("ledger.csv")
}

fn wal_config(dir: PathBuf) -> WalConfig {
    WalConfig {
        group_commit_window: Duration::from_millis(1),
        group_commit_max_batch: 4,
        ..WalConfig::new(dir)
    }
}

fn new_store(dir: &Path, cap: usize) -> Arc<BufferedRecordPageStore> {
    let io: Arc<dyn PageIo> =
        Arc::new(PosixPageIo::open_or_create(dir.join("record.store")).expect("open page io"));
    let pools = Arc::new(PerTenantBufferPool::with_config(
        io,
        PerTenantBufferPoolConfig {
            frames_per_tenant: 32,
            write_fraction: 0.0,
        },
    ));
    Arc::new(BufferedRecordPageStore::with_cache_cap(pools, cap))
}

/// Child workload: continuous-eviction-pressure commit loop. Exits
/// `WORKLOAD_CLEAN_EXIT_CODE` if it completes without being SIGKILL'd
/// (the crash window is tuned to make that rare, but the harness must
/// distinguish the two cases).
fn eviction_drain_workload(arg: &str) -> i32 {
    let dir = PathBuf::from(arg);
    let wal_dir = dir.join("wal");
    std::fs::create_dir_all(&wal_dir).expect("create wal dir");
    let store = new_store(&dir, CACHE_CAP);
    let dpt = Arc::new(DirtyPageTable::new());
    let props_target: Arc<dyn PageFlushTarget> = store.clone();
    let records_target: Arc<dyn PageFlushTarget> = store.clone();
    let checkpointer = Arc::new(WriteBehindCheckpointer::new(
        dpt.clone(),
        props_target,
        records_target,
    ));
    store.attach_m6_dirty_page_table(dpt.clone());
    store.attach_m6_checkpointer(checkpointer);

    let writer = WalWriter::spawn(wal_config(wal_dir)).expect("spawn wal writer");
    let handle = writer.handle();

    let mut ledger = std::fs::File::create(ledger_path(&dir)).expect("create ledger");
    // Second ledger: every page_no the WORKLOAD'S OWN store observed as
    // evicted (`is_evicted(pid) == true`) at some point. This is the
    // decisive, WAL-fallback-proof oracle: a page recorded here is
    // claimed by the checkpointer handshake to have a DURABLE HOME
    // WRITE, independent of whatever the WAL still holds (a real
    // checkpoint eventually prunes the WAL past a page's home write —
    // this ledger is what proves the disk home, not WAL replay, is
    // what must be correct for these specific pages).
    let mut evicted_ledger =
        std::fs::File::create(dir.join("evicted.csv")).expect("create evicted ledger");
    let mut previously_evicted = std::collections::HashSet::new();

    for i in 0..TOTAL_PAGES {
        let lsn = i + 1;
        let pid = PageId::new(i);
        let byte = (i as u8).wrapping_mul(53).wrapping_add(7);
        store
            .install_fresh(pid, PageType::Node, TenantId::DEFAULT)
            .expect("install_fresh");
        {
            let latch = RecordPageBackend::latch_for_tenant(store.as_ref(), TenantId::DEFAULT, pid)
                .expect("latch_for_tenant");
            latch.write().as_mut()[PAGE_SIZE - 1] = byte;
        }
        // Real WAL fsync BEFORE the ledger write — the ledger only ever
        // records a commit whose delta is already durable, matching
        // production's install-after-durability law.
        handle
            .append(
                WalRecordType::PutNode,
                i + 1,
                lsn as i64,
                TenantId::DEFAULT,
                vec![byte],
            )
            .expect("wal append");
        dpt.mark_dirty(
            DirtyPageKey {
                tenant_id: TenantId::DEFAULT,
                store_id: STORE_RECORD,
                page_no: pid.raw(),
            },
            Lsn::new(lsn),
        );

        // Continuous eviction pressure: MECH-E1..E8 fires on essentially
        // every commit given the tiny cap.
        let _ = store.evict_for_capacity(CACHE_CAP);

        // Ledger append AFTER the WAL fsync + DPT mark: this is the
        // parent's pre-crash ground truth. fsync the ledger file itself
        // so a SIGKILL right after this line still leaves the ledger
        // entry durable (the OS page cache alone is not enough across a
        // SIGKILL of THIS process, but `sync_all` forces it to the
        // actual disk before the next iteration proceeds).
        writeln!(ledger, "{i},{byte}").expect("ledger write");
        ledger.flush().expect("ledger flush");
        ledger.sync_all().expect("ledger fsync");

        // Record any page (among all installed so far) newly observed
        // as evicted — the checkpointer's durable-home-write claim.
        for j in 0..=i {
            if !previously_evicted.contains(&j) && store.is_evicted(PageId::new(j)) {
                previously_evicted.insert(j);
                writeln!(evicted_ledger, "{j}").expect("evicted ledger write");
            }
        }
        evicted_ledger.flush().expect("evicted ledger flush");
        evicted_ledger.sync_all().expect("evicted ledger fsync");

        std::thread::sleep(SLEEP_BETWEEN_COMMITS);
    }

    writer.shutdown().ok();
    WORKLOAD_CLEAN_EXIT_CODE
}

fn register_workloads_once() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        SubprocessWorkloadRegistry::register(WORKLOAD_NAME, eviction_drain_workload);
    });
}

fn dispatch_if_subprocess() {
    register_workloads_once();
    maybe_dispatch_subprocess_workload();
}

/// Must be a non-ignored test so the child subprocess (which passes
/// `--exact aaaa_subprocess_dispatcher_router`) always dispatches.
#[test]
fn aaaa_subprocess_dispatcher_router() {
    dispatch_if_subprocess();
}

fn read_ledger(path: &Path) -> Vec<(u64, u8)> {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    contents
        .lines()
        .filter_map(|line| {
            let (a, b) = line.split_once(',')?;
            Some((a.parse().ok()?, b.parse().ok()?))
        })
        .collect()
}

#[test]
#[ignore = "M6.1 real-SIGKILL eviction-drain gate; gated by \
            ARCGRAPH_M6_EVICT_SIGKILL_RUN=1; ~1-3s wall; panics if \
            neither ARCGRAPH_M6_EVICT_SIGKILL_RUN=1 nor \
            ARCGRAPH_M6_EVICT_SIGKILL_SKIP_OK=1 is set — see \
            feedback_test_env_gate_panic_by_default.md"]
fn eviction_drain_survives_real_sigkill() {
    dispatch_if_subprocess();

    let run = std::env::var(RUN_ENV).ok().as_deref() == Some("1");
    let skip_ok = std::env::var(SKIP_ENV).is_ok();
    if !run {
        if skip_ok {
            eprintln!(
                "eviction_drain_survives_real_sigkill: SKIPPING (opt-in via \
                 {SKIP_ENV}=1) — set {RUN_ENV}=1 to run the real SIGKILL gate instead"
            );
            return;
        }
        panic!(
            "eviction_drain_survives_real_sigkill: required env flag \
             missing. Set {RUN_ENV}=1 to run this real-SIGKILL gate, or \
             {SKIP_ENV}=1 to explicitly opt out (hostile-env / no \
             subprocess-fork support). Silent skip is not permitted \
             per feedback_test_env_gate_panic_by_default.md."
        );
    }

    let dir = tempfile::tempdir().unwrap();
    let record = run_with_crash_window_via_dispatcher(
        WORKLOAD_NAME,
        dir.path(),
        DISPATCHER_TEST,
        CRASH_WINDOW,
    )
    .expect("spawn + crash-window child");

    assert!(
        !record.exited_cleanly(),
        "workload exited cleanly before the crash window fired — the \
         harness is mistimed (workload too short / window too long); \
         re-tune SLEEP_BETWEEN_COMMITS or CRASH_WINDOW"
    );
    #[cfg(unix)]
    assert!(
        record.was_sigkilled(),
        "child did not die by SIGKILL as expected: {:?}",
        record.exit_status
    );

    // Recovery: a FRESH store bound to the SAME on-disk home file — no
    // in-process state survives a SIGKILL, so this is a genuine
    // durable-state check, not a graceful-shutdown one.
    let ledger = read_ledger(&ledger_path(dir.path()));
    assert!(
        !ledger.is_empty(),
        "ledger is empty — SIGKILL fired before the first commit; \
         widen CRASH_WINDOW or shorten SLEEP_BETWEEN_COMMITS"
    );
    eprintln!(
        "eviction_drain_survives_real_sigkill: {} pre-crash ledger entries, \
         killed after {:?}",
        ledger.len(),
        record.elapsed_to_kill
    );

    // Real recovery = disk home (for pages a checkpointer flush already
    // made durable) UNION WAL replay (for pages still resident-only at
    // SIGKILL time, whose durable delta is the WAL record itself — the
    // M3 install-after-durability law: dirty ⇒ WAL-durable, ALWAYS, even
    // if the store-file home write hasn't happened yet). This mirrors
    // what `redo::apply_recovery_delta` does in production; this
    // harness applies the (deliberately minimal) single-byte physical
    // delta this workload emits directly, since pulling in the full v9
    // commit-bundle/redo machinery is out of scope for this gate (the
    // dedicated WAL-replay gates cover that machinery; this gate's job
    // is proving EVICTION doesn't drop a page ahead of either durability
    // path).
    let wal_records = arcgraph_storage::wal::WalRecoveryReader::open(dir.path().join("wal"))
        .expect("open wal for recovery")
        .collect_all()
        .expect("collect wal records");
    let mut wal_byte_for_page: std::collections::HashMap<u64, u8> =
        std::collections::HashMap::new();
    for record in &wal_records {
        if record.record_type == WalRecordType::PutNode {
            // txn_id carries `page_no + 1` (see the workload's `.append`
            // call) — recovers the page identity without depending on
            // payload framing beyond the single byte this harness needs.
            let page_no = record.txn_id.saturating_sub(1);
            if let Some(byte) = record.payload.first().copied() {
                wal_byte_for_page.insert(page_no, byte);
            }
        }
    }

    let recovered = new_store(dir.path(), CACHE_CAP);
    let mut lost = Vec::new();
    for &(page_no, expected_byte) in &ledger {
        let pid = PageId::new(page_no);
        recovered.register_home_page(pid, TenantId::DEFAULT);
        let disk_byte = match recovered.fault_in(pid) {
            Ok(()) => {
                let latch = recovered.latch(pid).expect("latch after fault_in");
                let observed = latch.read().as_ref()[PAGE_SIZE - 1];
                Some(observed)
            }
            Err(_) => None,
        };
        let effective = match disk_byte {
            Some(byte) if byte == expected_byte => Some(byte),
            // Disk image absent or stale: the WAL record is the durable
            // source of truth for a page that never reached a
            // checkpointer home write before SIGKILL — apply it exactly
            // as production redo would.
            _ => wal_byte_for_page.get(&page_no).copied(),
        };
        match effective {
            Some(byte) if byte == expected_byte => {}
            other => lost.push((page_no, expected_byte, other)),
        }
    }
    assert!(
        lost.is_empty(),
        "eviction-drain SIGKILL recovery LOST or CORRUPTED {} of {} ledgered \
         pages (page_no, expected, effective-recovered): {:?} — neither the \
         disk home NOR the WAL record recovered the committed byte, which \
         means the ledgered commit's durability claim (WAL fsync succeeded) \
         was false — a genuine INV-M6.2 violation",
        lost.len(),
        ledger.len(),
        lost
    );

    // THE decisive, WAL-fallback-proof check: every page the workload's
    // OWN store observed as evicted (`evicted.csv`) MUST have a
    // byte-correct DISK HOME directly — no WAL replay credited. This is
    // what actually isolates MECH-E2/E3 (the checkpointer handshake)
    // from the WAL's own (legitimate, but separate) durability
    // contribution: a real checkpoint eventually prunes the WAL past a
    // page's durable home write, so "the WAL still has it" is not a
    // durability guarantee for a page the checkpointer already claimed
    // to have homed.
    let expected_by_page: std::collections::HashMap<u64, u8> = ledger.iter().copied().collect();
    let evicted_pages = read_ledger_single_column(&dir.path().join("evicted.csv"));
    assert!(
        !evicted_pages.is_empty(),
        "evicted.csv is empty — the tiny cache cap ({CACHE_CAP}) never forced \
         a real eviction before SIGKILL; widen TOTAL_PAGES or shrink CACHE_CAP \
         so this decisive leg is exercised"
    );
    let mut disk_home_wrong = Vec::new();
    for &page_no in &evicted_pages {
        let Some(&expected_byte) = expected_by_page.get(&page_no) else {
            continue;
        };
        let io: Arc<dyn PageIo> = Arc::new(
            PosixPageIo::open(dir.path().join("record.store")).expect("reopen disk file directly"),
        );
        let mut buf = [0u8; PAGE_SIZE];
        match io.read_page(PageId::new(page_no), &mut buf) {
            Ok(()) if buf[PAGE_SIZE - 1] == expected_byte => {}
            Ok(()) => disk_home_wrong.push((page_no, expected_byte, Some(buf[PAGE_SIZE - 1]))),
            Err(_) => disk_home_wrong.push((page_no, expected_byte, None)),
        }
    }
    assert!(
        disk_home_wrong.is_empty(),
        "MECH-E2/E3 violation: {} of {} evicted pages have a STALE or \
         UNREADABLE disk home (page_no, expected, observed): {:?} — the \
         checkpointer's durable-home-write handshake did not actually make \
         the byte durable before the frame was reclaimed",
        disk_home_wrong.len(),
        evicted_pages.len(),
        disk_home_wrong
    );
}

fn read_ledger_single_column(path: &Path) -> Vec<u64> {
    let Ok(contents) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    contents
        .lines()
        .filter_map(|line| line.parse().ok())
        .collect()
}
