//! Test-only helpers.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};

static COUNTER: AtomicU32 = AtomicU32::new(0);

/// A unique temporary directory that is removed on drop. Avoids pulling in an
/// external tempdir crate for the few I/O tests in this crate.
pub struct TempRunDir {
    path: PathBuf,
}

impl TempRunDir {
    /// Creates a fresh, uniquely-named directory under the system temp dir.
    pub fn new() -> Self {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("emry-store-test-{}-{n}", std::process::id()));
        std::fs::create_dir_all(&path).expect("create temp run dir");
        Self { path }
    }

    /// Path to the directory.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempRunDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}
