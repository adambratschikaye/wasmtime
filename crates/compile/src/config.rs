use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use target_lexicon::Architecture;
use wasmtime_environ::CacheStore;

/// Possible Compilation strategies for a wasm module.
///
/// This is used as an argument to the [`Config::strategy`] method.
#[non_exhaustive]
#[derive(PartialEq, Eq, Clone, Debug, Copy)]
pub enum Strategy {
    /// An indicator that the compilation strategy should be automatically
    /// selected.
    ///
    /// This is generally what you want for most projects and indicates that the
    /// `wasmtime` crate itself should make the decision about what the best
    /// code generator for a wasm module is.
    ///
    /// Currently this always defaults to Cranelift, but the default value may
    /// change over time.
    Auto,

    /// Currently the default backend, Cranelift aims to be a reasonably fast
    /// code generator which generates high quality machine code.
    Cranelift,

    /// A baseline compiler for WebAssembly, currently under active development and not ready for
    /// production applications.
    Winch,
}

/// User-provided configuration for the compiler.
#[derive(Debug, Clone)]
pub struct CompilerConfig {
    pub strategy: Strategy,
    pub target: Option<target_lexicon::Triple>,
    pub settings: HashMap<String, String>,
    pub flags: HashSet<String>,
    pub cache_store: Option<Arc<dyn CacheStore>>,
    pub clif_dir: Option<std::path::PathBuf>,
    pub wmemcheck: bool,
}

impl CompilerConfig {
    fn new(strategy: Strategy) -> Self {
        Self {
            strategy,
            target: None,
            settings: HashMap::new(),
            flags: HashSet::new(),
            cache_store: None,
            clif_dir: None,
            wmemcheck: false,
        }
    }

    /// Ensures that the key is not set or equals to the given value.
    /// If the key is not set, it will be set to the given value.
    ///
    /// # Returns
    ///
    /// Returns true if successfully set or already had the given setting
    /// value, or false if the setting was explicitly set to something
    /// else previously.
    pub(crate) fn ensure_setting_unset_or_given(&mut self, k: &str, v: &str) -> bool {
        if let Some(value) = self.settings.get(k) {
            if value != v {
                return false;
            }
        } else {
            self.settings.insert(k.to_string(), v.to_string());
        }
        true
    }
}

impl Default for CompilerConfig {
    fn default() -> Self {
        Self::new(Strategy::Auto)
    }
}

pub fn probestack_supported(arch: Architecture) -> bool {
    matches!(
        arch,
        Architecture::X86_64 | Architecture::Aarch64(_) | Architecture::Riscv64(_)
    )
}
