use crate::profiling::ProfilingAgent;
use crate::signatures::SignatureRegistry;
use crate::unwind::UnwindRegistration;
use crate::{Config, Module};
use anyhow::{Context, Result};
use object::write::{Object, StandardSegment};
use object::SectionKind;
use once_cell::sync::OnceCell;
#[cfg(feature = "parallel-compilation")]
use rayon::prelude::*;
use std::mem::ManuallyDrop;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
#[cfg(feature = "cache")]
use wasmtime_cache::CacheConfig;
use wasmtime_environ::{
    obj, DefinedFuncIndex, FuncIndex, FunctionLoc, PrimaryMap, SignatureIndex, StackMapInformation,
    WasmFunctionInfo,
};
use wasmtime_environ::{FlagValue, ObjectKind};
use wasmtime_jit::{CodeMemory, CompiledFunctionInfo, CompiledModuleInfo, LibCalls};
use wasmtime_runtime::{libcalls::relocs, CompiledModuleIdAllocator, InstanceAllocator, MmapVec};
use wasmtime_runtime::{CompiledModuleId, GdbJitImageRegistration};

mod serialization;

pub(crate) const LIBCALLS: LibCalls = LibCalls {
    floorf32: relocs::floorf32 as usize,
    floorf64: relocs::floorf64 as usize,
    nearestf32: relocs::nearestf32 as usize,
    nearestf64: relocs::nearestf64 as usize,
    ceilf32: relocs::ceilf32 as usize,
    ceilf64: relocs::ceilf64 as usize,
    truncf32: relocs::truncf32 as usize,
    truncf64: relocs::truncf64 as usize,
    fmaf32: relocs::fmaf32 as usize,
    fmaf64: relocs::fmaf64 as usize,
    x86_pshufb: relocs::x86_pshufb as usize,
};

/// An `Engine` which is a global context for compilation and management of wasm
/// modules.
///
/// An engine can be safely shared across threads and is a cheap cloneable
/// handle to the actual engine. The engine itself will be deallocated once all
/// references to it have gone away.
///
/// Engines store global configuration preferences such as compilation settings,
/// enabled features, etc. You'll likely only need at most one of these for a
/// program.
///
/// ## Engines and `Clone`
///
/// Using `clone` on an `Engine` is a cheap operation. It will not create an
/// entirely new engine, but rather just a new reference to the existing engine.
/// In other words it's a shallow copy, not a deep copy.
///
/// ## Engines and `Default`
///
/// You can create an engine with default configuration settings using
/// `Engine::default()`. Be sure to consult the documentation of [`Config`] for
/// default settings.
#[derive(Clone)]
pub struct Engine {
    inner: Arc<EngineInner>,
}

struct EngineInner {
    config: Config,
    #[cfg(any(feature = "cranelift", feature = "winch"))]
    compiler: Box<dyn wasmtime_environ::Compiler>,
    allocator: Box<dyn InstanceAllocator + Send + Sync>,
    profiler: Box<dyn ProfilingAgent>,
    signatures: SignatureRegistry,
    epoch: AtomicU64,
    unique_id_allocator: CompiledModuleIdAllocator,

    // One-time check of whether the compiler's settings, if present, are
    // compatible with the native host.
    compatible_with_native_host: OnceCell<Result<(), String>>,
}

impl Engine {
    /// Creates a new [`Engine`] with the specified compilation and
    /// configuration settings.
    ///
    /// # Errors
    ///
    /// This method can fail if the `config` is invalid or some
    /// configurations are incompatible.
    ///
    /// For example, feature `reference_types` will need to set
    /// the compiler setting `enable_safepoints` and `unwind_info`
    /// to `true`, but explicitly disable these two compiler settings
    /// will cause errors.
    pub fn new(config: &Config) -> Result<Engine> {
        // Ensure that wasmtime_runtime's signal handlers are configured. This
        // is the per-program initialization required for handling traps, such
        // as configuring signals, vectored exception handlers, etc.
        wasmtime_runtime::init_traps(crate::module::is_wasm_trap_pc, config.macos_use_mach_ports);
        #[cfg(feature = "debug-builtins")]
        wasmtime_runtime::debug_builtins::ensure_exported();

        let registry = SignatureRegistry::new();
        let config = config.clone();
        config.validate()?;

        #[cfg(any(feature = "cranelift", feature = "winch"))]
        let (config, compiler) = config.build_compiler()?;

        let allocator = config.build_allocator()?;
        let profiler = config.build_profiler()?;

        Ok(Engine {
            inner: Arc::new(EngineInner {
                #[cfg(any(feature = "cranelift", feature = "winch"))]
                compiler,
                config,
                allocator,
                profiler,
                signatures: registry,
                epoch: AtomicU64::new(0),
                unique_id_allocator: CompiledModuleIdAllocator::new(),
                compatible_with_native_host: OnceCell::new(),
            }),
        })
    }

    /// Eagerly initialize thread-local functionality shared by all [`Engine`]s.
    ///
    /// Wasmtime's implementation on some platforms may involve per-thread
    /// setup that needs to happen whenever WebAssembly is invoked. This setup
    /// can take on the order of a few hundred microseconds, whereas the
    /// overhead of calling WebAssembly is otherwise on the order of a few
    /// nanoseconds. This setup cost is paid once per-OS-thread. If your
    /// application is sensitive to the latencies of WebAssembly function
    /// calls, even those that happen first on a thread, then this function
    /// can be used to improve the consistency of each call into WebAssembly
    /// by explicitly frontloading the cost of the one-time setup per-thread.
    ///
    /// Note that this function is not required to be called in any embedding.
    /// Wasmtime will automatically initialize thread-local-state as necessary
    /// on calls into WebAssembly. This is provided for use cases where the
    /// latency of WebAssembly calls are extra-important, which is not
    /// necessarily true of all embeddings.
    pub fn tls_eager_initialize() {
        wasmtime_runtime::tls_eager_initialize();
    }

    /// Returns the configuration settings that this engine is using.
    #[inline]
    pub fn config(&self) -> &Config {
        &self.inner.config
    }

    #[cfg(any(feature = "cranelift", feature = "winch"))]
    pub(crate) fn compiler(&self) -> &dyn wasmtime_environ::Compiler {
        &*self.inner.compiler
    }

    pub(crate) fn allocator(&self) -> &dyn InstanceAllocator {
        self.inner.allocator.as_ref()
    }

    pub(crate) fn profiler(&self) -> &dyn ProfilingAgent {
        self.inner.profiler.as_ref()
    }

    #[cfg(feature = "cache")]
    pub(crate) fn cache_config(&self) -> &CacheConfig {
        &self.config().cache_config
    }

    /// Returns whether the engine `a` and `b` refer to the same configuration.
    pub fn same(a: &Engine, b: &Engine) -> bool {
        Arc::ptr_eq(&a.inner, &b.inner)
    }

    pub(crate) fn signatures(&self) -> &SignatureRegistry {
        &self.inner.signatures
    }

    pub(crate) fn epoch_counter(&self) -> &AtomicU64 {
        &self.inner.epoch
    }

    pub(crate) fn current_epoch(&self) -> u64 {
        self.epoch_counter().load(Ordering::Relaxed)
    }

    /// Increments the epoch.
    ///
    /// When using epoch-based interruption, currently-executing Wasm
    /// code within this engine will trap or yield "soon" when the
    /// epoch deadline is reached or exceeded. (The configuration, and
    /// the deadline, are set on the `Store`.) The intent of the
    /// design is for this method to be called by the embedder at some
    /// regular cadence, for example by a thread that wakes up at some
    /// interval, or by a signal handler.
    ///
    /// See [`Config::epoch_interruption`](crate::Config::epoch_interruption)
    /// for an introduction to epoch-based interruption and pointers
    /// to the other relevant methods.
    ///
    /// ## Signal Safety
    ///
    /// This method is signal-safe: it does not make any syscalls, and
    /// performs only an atomic increment to the epoch value in
    /// memory.
    pub fn increment_epoch(&self) {
        self.inner.epoch.fetch_add(1, Ordering::Relaxed);
    }

    pub(crate) fn unique_id_allocator(&self) -> &CompiledModuleIdAllocator {
        &self.inner.unique_id_allocator
    }

    /// Ahead-of-time (AOT) compiles a WebAssembly module.
    ///
    /// The `bytes` provided must be in one of two formats:
    ///
    /// * A [binary-encoded][binary] WebAssembly module. This is always supported.
    /// * A [text-encoded][text] instance of the WebAssembly text format.
    ///   This is only supported when the `wat` feature of this crate is enabled.
    ///   If this is supplied then the text format will be parsed before validation.
    ///   Note that the `wat` feature is enabled by default.
    ///
    /// This method may be used to compile a module for use with a different target
    /// host. The output of this method may be used with
    /// [`Module::deserialize`](crate::Module::deserialize) on hosts compatible
    /// with the [`Config`] associated with this [`Engine`].
    ///
    /// The output of this method is safe to send to another host machine for later
    /// execution. As the output is already a compiled module, translation and code
    /// generation will be skipped and this will improve the performance of constructing
    /// a [`Module`](crate::Module) from the output of this method.
    ///
    /// [binary]: https://webassembly.github.io/spec/core/binary/index.html
    /// [text]: https://webassembly.github.io/spec/core/text/index.html
    #[cfg(any(feature = "cranelift", feature = "winch"))]
    #[cfg_attr(nightlydoc, doc(cfg(any(feature = "cranelift", feature = "winch"))))]
    pub fn precompile_module(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        #[cfg(feature = "wat")]
        let bytes = wat::parse_bytes(&bytes)?;
        let (mmap, _) = crate::Module::build_artifacts(self, &bytes)?;
        Ok(mmap.to_vec())
    }

    /// Same as [`Engine::precompile_module`] except for a
    /// [`Component`](crate::component::Component)
    #[cfg(any(feature = "cranelift", feature = "winch"))]
    #[cfg_attr(nightlydoc, doc(cfg(any(feature = "cranelift", feature = "winch"))))]
    #[cfg(feature = "component-model")]
    #[cfg_attr(nightlydoc, doc(cfg(feature = "component-model")))]
    pub fn precompile_component(&self, bytes: &[u8]) -> Result<Vec<u8>> {
        #[cfg(feature = "wat")]
        let bytes = wat::parse_bytes(&bytes)?;
        let (mmap, _) = crate::component::Component::build_artifacts(self, &bytes)?;
        Ok(mmap.to_vec())
    }

    /// Returns a [`std::hash::Hash`] that can be used to check precompiled WebAssembly compatibility.
    ///
    /// The outputs of [`Engine::precompile_module`] and [`Engine::precompile_component`]
    /// are compatible with a different [`Engine`] instance only if the two engines use
    /// compatible [`Config`]s. If this Hash matches between two [`Engine`]s then binaries
    /// from one are guaranteed to deserialize in the other.
    #[cfg(any(feature = "cranelift", feature = "winch"))]
    #[cfg_attr(nightlydoc, doc(cfg(feature = "cranelift")))] // see build.rs
    pub fn precompile_compatibility_hash(&self) -> impl std::hash::Hash + '_ {
        crate::module::HashedEngineCompileEnv(self)
    }

    pub(crate) fn run_maybe_parallel<
        A: Send,
        B: Send,
        E: Send,
        F: Fn(A) -> Result<B, E> + Send + Sync,
    >(
        &self,
        input: Vec<A>,
        f: F,
    ) -> Result<Vec<B>, E> {
        if self.config().parallel_compilation {
            #[cfg(feature = "parallel-compilation")]
            return input
                .into_par_iter()
                .map(|a| f(a))
                .collect::<Result<Vec<B>, E>>();
        }

        // In case the parallel-compilation feature is disabled or the parallel_compilation config
        // was turned off dynamically fallback to the non-parallel version.
        input
            .into_iter()
            .map(|a| f(a))
            .collect::<Result<Vec<B>, E>>()
    }

    /// Executes `f1` and `f2` in parallel if parallel compilation is enabled at
    /// both runtime and compile time, otherwise runs them synchronously.
    #[allow(dead_code)] // only used for the component-model feature right now
    pub(crate) fn join_maybe_parallel<T, U>(
        &self,
        f1: impl FnOnce() -> T + Send,
        f2: impl FnOnce() -> U + Send,
    ) -> (T, U)
    where
        T: Send,
        U: Send,
    {
        if self.config().parallel_compilation {
            #[cfg(feature = "parallel-compilation")]
            return rayon::join(f1, f2);
        }
        (f1(), f2())
    }

    /// Returns the target triple which this engine is compiling code for
    /// and/or running code for.
    pub(crate) fn target(&self) -> target_lexicon::Triple {
        // If a compiler is configured, use that target.
        #[cfg(any(feature = "cranelift", feature = "winch"))]
        return self.compiler().triple().clone();

        // ... otherwise it's the native target
        #[cfg(not(any(feature = "cranelift", feature = "winch")))]
        return target_lexicon::Triple::host();
    }

    /// Verify that this engine's configuration is compatible with loading
    /// modules onto the native host platform.
    ///
    /// This method is used as part of `Module::new` to ensure that this
    /// engine can indeed load modules for the configured compiler (if any).
    /// Note that if cranelift is disabled this trivially returns `Ok` because
    /// loaded serialized modules are checked separately.
    pub(crate) fn check_compatible_with_native_host(&self) -> Result<()> {
        self.inner
            .compatible_with_native_host
            .get_or_init(|| self._check_compatible_with_native_host())
            .clone()
            .map_err(anyhow::Error::msg)
    }

    fn _check_compatible_with_native_host(&self) -> Result<(), String> {
        #[cfg(any(feature = "cranelift", feature = "winch"))]
        {
            let compiler = self.compiler();

            // Check to see that the config's target matches the host
            let target = compiler.triple();
            if *target != target_lexicon::Triple::host() {
                return Err(format!(
                    "target '{}' specified in the configuration does not match the host",
                    target
                ));
            }

            // Also double-check all compiler settings
            for (key, value) in compiler.flags().iter() {
                self.check_compatible_with_shared_flag(key, value)?;
            }
            for (key, value) in compiler.isa_flags().iter() {
                self.check_compatible_with_isa_flag(key, value)?;
            }
        }
        Ok(())
    }

    /// Checks to see whether the "shared flag", something enabled for
    /// individual compilers, is compatible with the native host platform.
    ///
    /// This is used both when validating an engine's compilation settings are
    /// compatible with the host as well as when deserializing modules from
    /// disk to ensure they're compatible with the current host.
    ///
    /// Note that most of the settings here are not configured by users that
    /// often. While theoretically possible via `Config` methods the more
    /// interesting flags are the ISA ones below. Typically the values here
    /// represent global configuration for wasm features. Settings here
    /// currently rely on the compiler informing us of all settings, including
    /// those disabled. Settings then fall in a few buckets:
    ///
    /// * Some settings must be enabled, such as `preserve_frame_pointers`.
    /// * Some settings must have a particular value, such as
    ///   `libcall_call_conv`.
    /// * Some settings do not matter as to their value, such as `opt_level`.
    pub(crate) fn check_compatible_with_shared_flag(
        &self,
        flag: &str,
        value: &FlagValue,
    ) -> Result<(), String> {
        let target = self.target();
        let ok = match flag {
            // These settings must all have be enabled, since their value
            // can affect the way the generated code performs or behaves at
            // runtime.
            "libcall_call_conv" => *value == FlagValue::Enum("isa_default".into()),
            "preserve_frame_pointers" => *value == FlagValue::Bool(true),
            "enable_probestack" => *value == FlagValue::Bool(crate::config::probestack_supported(target.architecture)),
            "probestack_strategy" => *value == FlagValue::Enum("inline".into()),

            // Features wasmtime doesn't use should all be disabled, since
            // otherwise if they are enabled it could change the behavior of
            // generated code.
            "enable_llvm_abi_extensions" => *value == FlagValue::Bool(false),
            "enable_pinned_reg" => *value == FlagValue::Bool(false),
            "use_colocated_libcalls" => *value == FlagValue::Bool(false),
            "use_pinned_reg_as_heap_base" => *value == FlagValue::Bool(false),

            // If reference types are enabled this must be enabled, otherwise
            // this setting can have any value.
            "enable_safepoints" => {
                if self.config().features.reference_types {
                    *value == FlagValue::Bool(true)
                } else {
                    return Ok(())
                }
            }

            // Windows requires unwind info as part of its ABI.
            "unwind_info" => {
                if target.operating_system == target_lexicon::OperatingSystem::Windows {
                    *value == FlagValue::Bool(true)
                } else {
                    return Ok(())
                }
            }

            // These settings don't affect the interface or functionality of
            // the module itself, so their configuration values shouldn't
            // matter.
            "enable_heap_access_spectre_mitigation"
            | "enable_table_access_spectre_mitigation"
            | "enable_nan_canonicalization"
            | "enable_jump_tables"
            | "enable_float"
            | "enable_verifier"
            | "enable_pcc"
            | "regalloc_checker"
            | "regalloc_verbose_logs"
            | "is_pic"
            | "bb_padding_log2_minus_one"
            | "machine_code_cfg_info"
            | "tls_model" // wasmtime doesn't use tls right now
            | "opt_level" // opt level doesn't change semantics
            | "enable_alias_analysis" // alias analysis-based opts don't change semantics
            | "probestack_func_adjusts_sp" // probestack above asserted disabled
            | "probestack_size_log2" // probestack above asserted disabled
            | "regalloc" // shouldn't change semantics
            | "enable_incremental_compilation_cache_checks" // shouldn't change semantics
            | "enable_atomics" => return Ok(()),

            // Everything else is unknown and needs to be added somewhere to
            // this list if encountered.
            _ => {
                return Err(format!("unknown shared setting {:?} configured to {:?}", flag, value))
            }
        };

        if !ok {
            return Err(format!(
                "setting {:?} is configured to {:?} which is not supported",
                flag, value,
            ));
        }
        Ok(())
    }

    /// Same as `check_compatible_with_native_host` except used for ISA-specific
    /// flags. This is used to test whether a configured ISA flag is indeed
    /// available on the host platform itself.
    pub(crate) fn check_compatible_with_isa_flag(
        &self,
        flag: &str,
        value: &FlagValue,
    ) -> Result<(), String> {
        match value {
            // ISA flags are used for things like CPU features, so if they're
            // disabled then it's compatible with the native host.
            FlagValue::Bool(false) => return Ok(()),

            // Fall through below where we test at runtime that features are
            // available.
            FlagValue::Bool(true) => {}

            // Only `bool` values are supported right now, other settings would
            // need more support here.
            _ => {
                return Err(format!(
                    "isa-specific feature {:?} configured to unknown value {:?}",
                    flag, value
                ))
            }
        }

        #[allow(unused_assignments)]
        let mut enabled = None;

        #[cfg(target_arch = "aarch64")]
        {
            enabled = match flag {
                "has_lse" => Some(std::arch::is_aarch64_feature_detected!("lse")),
                // No effect on its own, but in order to simplify the code on a
                // platform without pointer authentication support we fail if
                // "has_pauth" is enabled, but "sign_return_address" is not.
                "has_pauth" => Some(std::arch::is_aarch64_feature_detected!("paca")),
                // No effect on its own.
                "sign_return_address_all" => Some(true),
                // The pointer authentication instructions act as a `NOP` when
                // unsupported (but keep in mind "has_pauth" as well), so it is
                // safe to enable them.
                "sign_return_address" => Some(true),
                // No effect on its own.
                "sign_return_address_with_bkey" => Some(true),
                // The `BTI` instruction acts as a `NOP` when unsupported, so it
                // is safe to enable it.
                "use_bti" => Some(true),
                // fall through to the very bottom to indicate that support is
                // not enabled to test whether this feature is enabled on the
                // host.
                _ => None,
            };
        }

        // There is no is_s390x_feature_detected macro yet, so for now
        // we use getauxval from the libc crate directly.
        #[cfg(all(target_arch = "s390x", target_os = "linux"))]
        {
            let v = unsafe { libc::getauxval(libc::AT_HWCAP) };
            const HWCAP_S390X_VXRS_EXT2: libc::c_ulong = 32768;

            enabled = match flag {
                // There is no separate HWCAP bit for mie2, so assume
                // that any machine with vxrs_ext2 also has mie2.
                "has_vxrs_ext2" | "has_mie2" => Some((v & HWCAP_S390X_VXRS_EXT2) != 0),
                // fall through to the very bottom to indicate that support is
                // not enabled to test whether this feature is enabled on the
                // host.
                _ => None,
            }
        }

        #[cfg(target_arch = "riscv64")]
        {
            enabled = match flag {
                // make sure `test_isa_flags_mismatch` test pass.
                "not_a_flag" => None,
                // due to `is_riscv64_feature_detected` is not stable.
                // we cannot use it.
                _ => Some(true),
            }
        }

        #[cfg(target_arch = "x86_64")]
        {
            enabled = match flag {
                "has_sse3" => Some(std::is_x86_feature_detected!("sse3")),
                "has_ssse3" => Some(std::is_x86_feature_detected!("ssse3")),
                "has_sse41" => Some(std::is_x86_feature_detected!("sse4.1")),
                "has_sse42" => Some(std::is_x86_feature_detected!("sse4.2")),
                "has_popcnt" => Some(std::is_x86_feature_detected!("popcnt")),
                "has_avx" => Some(std::is_x86_feature_detected!("avx")),
                "has_avx2" => Some(std::is_x86_feature_detected!("avx2")),
                "has_fma" => Some(std::is_x86_feature_detected!("fma")),
                "has_bmi1" => Some(std::is_x86_feature_detected!("bmi1")),
                "has_bmi2" => Some(std::is_x86_feature_detected!("bmi2")),
                "has_avx512bitalg" => Some(std::is_x86_feature_detected!("avx512bitalg")),
                "has_avx512dq" => Some(std::is_x86_feature_detected!("avx512dq")),
                "has_avx512f" => Some(std::is_x86_feature_detected!("avx512f")),
                "has_avx512vl" => Some(std::is_x86_feature_detected!("avx512vl")),
                "has_avx512vbmi" => Some(std::is_x86_feature_detected!("avx512vbmi")),
                "has_lzcnt" => Some(std::is_x86_feature_detected!("lzcnt")),

                // fall through to the very bottom to indicate that support is
                // not enabled to test whether this feature is enabled on the
                // host.
                _ => None,
            };
        }

        match enabled {
            Some(true) => return Ok(()),
            Some(false) => {
                return Err(format!(
                    "compilation setting {:?} is enabled, but not available on the host",
                    flag
                ))
            }
            // fall through
            None => {}
        }

        Err(format!(
            "cannot test if target-specific flag {:?} is available at runtime",
            flag
        ))
    }

    #[cfg(any(feature = "cranelift", feature = "winch"))]
    pub(crate) fn append_compiler_info(&self, obj: &mut Object<'_>) {
        serialization::append_compiler_info(self, obj);
    }

    #[cfg(any(feature = "cranelift", feature = "winch"))]
    pub(crate) fn append_bti(&self, obj: &mut Object<'_>) {
        let section = obj.add_section(
            obj.segment_name(StandardSegment::Data).to_vec(),
            obj::ELF_WASM_BTI.as_bytes().to_vec(),
            SectionKind::ReadOnlyData,
        );
        let contents = if self.compiler().is_branch_protection_enabled() {
            1
        } else {
            0
        };
        obj.append_section_data(section, &[contents], 1);
    }

    /// Loads a `CodeMemory` from the specified in-memory slice, copying it to a
    /// uniquely owned mmap.
    ///
    /// The `expected` marker here is whether the bytes are expected to be a
    /// precompiled module or a component.
    pub(crate) fn load_code_bytes(
        &self,
        bytes: &[u8],
        expected: ObjectKind,
    ) -> Result<Arc<CodeMemory>> {
        self.load_code(MmapVec::from_slice(bytes)?, expected)
    }

    /// Like `load_code_bytes`, but creates a mmap from a file on disk.
    pub(crate) fn load_code_file(
        &self,
        path: &Path,
        expected: ObjectKind,
    ) -> Result<Arc<CodeMemory>> {
        self.load_code(
            MmapVec::from_file(path).with_context(|| {
                format!("failed to create file mapping for: {}", path.display())
            })?,
            expected,
        )
    }

    pub(crate) fn load_code(&self, mmap: MmapVec, expected: ObjectKind) -> Result<Arc<CodeMemory>> {
        serialization::check_compatible(self, &mmap, expected)?;
        let mut code = CodeMemory::new(mmap)?;
        code.publish(LIBCALLS)?;
        Ok(Arc::new(code))
    }

    /// Detects whether the bytes provided are a precompiled object produced by
    /// Wasmtime.
    ///
    /// This function will inspect the header of `bytes` to determine if it
    /// looks like a precompiled core wasm module or a precompiled component.
    /// This does not validate the full structure or guarantee that
    /// deserialization will succeed, instead it helps higher-levels of the
    /// stack make a decision about what to do next when presented with the
    /// `bytes` as an input module.
    ///
    /// If the `bytes` looks like a precompiled object previously produced by
    /// [`Module::serialize`](crate::Module::serialize),
    /// [`Component::serialize`](crate::component::Component::serialize),
    /// [`Engine::precompile_module`], or [`Engine::precompile_component`], then
    /// this will return `Some(...)` indicating so. Otherwise `None` is
    /// returned.
    pub fn detect_precompiled(&self, bytes: &[u8]) -> Option<Precompiled> {
        serialization::detect_precompiled_bytes(bytes)
    }

    /// Like [`Engine::detect_precompiled`], but performs the detection on a file.
    pub fn detect_precompiled_file(&self, path: impl AsRef<Path>) -> Result<Option<Precompiled>> {
        serialization::detect_precompiled_file(path)
    }
}

impl Default for Engine {
    fn default() -> Engine {
        Engine::new(&Config::default()).unwrap()
    }
}

/// Return value from the [`Engine::detect_precompiled`] API.
#[derive(PartialEq, Eq, Copy, Clone, Debug)]
pub enum Precompiled {
    /// The input bytes look like a precompiled core wasm module.
    Module,
    /// The input bytes look like a precompiled wasm component.
    Component,
}

struct MmapCodeMemory {
    memory: ManuallyDrop<CodeMemory<MmapVec>>,
    unwind_registration: ManuallyDrop<Option<UnwindRegistration>>,
    published: bool,
}

impl Drop for MmapCodeMemory {
    fn drop(&mut self) {
        // Drop `unwind_registration` before `self.memory`
        unsafe {
            ManuallyDrop::drop(&mut self.unwind_registration);
            ManuallyDrop::drop(&mut self.memory);
        }
    }
}

impl MmapCodeMemory {
    unsafe fn register_unwind_info(&mut self) -> Result<()> {
        if self.memory.unwind().len() == 0 {
            return Ok(());
        }
        let text = self.memory.text();
        let unwind_info = self.memory.unwind();
        let registration =
            UnwindRegistration::new(text.as_ptr(), unwind_info.as_ptr(), unwind_info.len())
                .context("failed to create unwind info registration")?;
        *self.unwind_registration = Some(registration);
        Ok(())
    }

    /// Publishes the internal ELF image to be ready for execution.
    ///
    /// This method can only be called once and will panic if called twice. This
    /// will parse the ELF image from the original `MmapVec` and do everything
    /// necessary to get it ready for execution, including:
    ///
    /// * Change page protections from read/write to read/execute.
    /// * Register unwinding information with the OS
    ///
    /// After this function executes all JIT code should be ready to execute.
    pub fn publish(&mut self, libcalls: LibCalls) -> Result<()> {
        assert!(!self.published);
        self.published = true;

        if self.memory.text().is_empty() {
            return Ok(());
        }

        // The unsafety here comes from a few things:
        //
        // * We're actually updating some page protections to executable memory.
        //
        // * We're registering unwinding information which relies on the
        //   correctness of the information in the first place. This applies to
        //   both the actual unwinding tables as well as the validity of the
        //   pointers we pass in itself.
        unsafe {
            // First, if necessary, apply relocations. This can happen for
            // things like libcalls which happen late in the lowering process
            // that don't go through the Wasm-based libcalls layer that's
            // indirected through the `VMContext`. Note that most modules won't
            // have relocations, so this typically doesn't do anything.
            self.memory.apply_relocations(libcalls)?;

            // Next freeze the contents of this image by making all of the
            // memory readonly. Nothing after this point should ever be modified
            // so commit everything. For a compiled-in-memory image this will
            // mean IPIs to evict writable mappings from other cores. For
            // loaded-from-disk images this shouldn't result in IPIs so long as
            // there weren't any relocations because nothing should have
            // otherwise written to the image at any point either.
            self.memory.mmap.make_readonly(0..self.mmap.len())?;

            let text = self.memory.text();

            // Clear the newly allocated code from cache if the processor requires it
            //
            // Do this before marking the memory as R+X, technically we should be able to do it after
            // but there are some CPU's that have had errata about doing this with read only memory.
            icache_coherence::clear_cache(text.as_ptr().cast(), text.len())
                .expect("Failed cache clear");

            // Switch the executable portion from readonly to read/execute.
            self.memory
                .mmap
                .make_executable(self.text.clone(), self.enable_branch_protection)
                .context("unable to make memory executable")?;

            // Flush any in-flight instructions from the pipeline
            icache_coherence::pipeline_flush_mt().expect("Failed pipeline flush");

            // With all our memory set up use the platform-specific
            // `UnwindRegistration` implementation to inform the general
            // runtime that there's unwinding information available for all
            // our just-published JIT functions.
            self.register_unwind_info()?;
        }

        Ok(())
    }
}

/// A compiled wasm module, ready to be instantiated.
pub struct CompiledModule {
    module: Arc<Module>,
    funcs: PrimaryMap<DefinedFuncIndex, CompiledFunctionInfo>,
    wasm_to_native_trampolines: Vec<(SignatureIndex, FunctionLoc)>,
    meta: Metadata,
    code_memory: Arc<CodeMemory>,
    dbg_jit_registration: Option<GdbJitImageRegistration>,
    /// A unique ID used to register this module with the engine.
    unique_id: CompiledModuleId,
    func_names: Vec<FunctionName>,
}

impl CompiledModule {
    /// Creates `CompiledModule` directly from a precompiled artifact.
    ///
    /// The `code_memory` argument is expected to be the result of a previous
    /// call to `ObjectBuilder::finish` above. This is an ELF image, at this
    /// time, which contains all necessary information to create a
    /// `CompiledModule` from a compilation.
    ///
    /// This method also takes `info`, an optionally-provided deserialization
    /// of the artifacts' compilation metadata section. If this information is
    /// not provided then the information will be
    /// deserialized from the image of the compilation artifacts. Otherwise it
    /// will be assumed to be what would otherwise happen if the section were
    /// to be deserialized.
    ///
    /// The `profiler` argument here is used to inform JIT profiling runtimes
    /// about new code that is loaded.
    pub fn from_artifacts(
        code_memory: Arc<CodeMemory>,
        info: CompiledModuleInfo,
        profiler: &dyn ProfilingAgent,
        id_allocator: &CompiledModuleIdAllocator,
    ) -> Result<Self> {
        let mut ret = Self {
            module: Arc::new(info.module),
            funcs: info.funcs,
            wasm_to_native_trampolines: info.wasm_to_native_trampolines,
            dbg_jit_registration: None,
            code_memory,
            meta: info.meta,
            unique_id: id_allocator.alloc(),
            func_names: info.func_names,
        };
        ret.register_debug_and_profiling(profiler)?;

        Ok(ret)
    }

    fn register_debug_and_profiling(&mut self, profiler: &dyn ProfilingAgent) -> Result<()> {
        if self.meta.native_debug_info_present {
            let text = self.text();
            let bytes = create_gdbjit_image(self.mmap().to_vec(), (text.as_ptr(), text.len()))
                .context("failed to create jit image for gdb")?;
            let reg = GdbJitImageRegistration::register(bytes);
            self.dbg_jit_registration = Some(reg);
        }
        profiler.register_module(&self.code_memory, &|addr| {
            let (idx, _) = self.func_by_text_offset(addr)?;
            let idx = self.module.func_index(idx);
            let name = self.func_name(idx)?;
            let mut demangled = String::new();
            wasmtime_jit::demangle_function_name(&mut demangled, name).unwrap();
            Some(demangled)
        });
        Ok(())
    }

    /// Get this module's unique ID. It is unique with respect to a
    /// single allocator (which is ordinarily held on a Wasm engine).
    pub fn unique_id(&self) -> CompiledModuleId {
        self.unique_id
    }

    /// Returns the underlying memory which contains the compiled module's
    /// image.
    pub fn mmap(&self) -> &MmapVec {
        self.code_memory.mmap()
    }

    /// Returns the underlying owned mmap of this compiled image.
    pub fn code_memory(&self) -> &Arc<CodeMemory> {
        &self.code_memory
    }

    /// Returns the text section of the ELF image for this compiled module.
    ///
    /// This memory should have the read/execute permissions.
    #[inline]
    pub fn text(&self) -> &[u8] {
        self.code_memory.text()
    }

    /// Return a reference-counting pointer to a module.
    pub fn module(&self) -> &Arc<Module> {
        &self.module
    }

    /// Looks up the `name` section name for the function index `idx`, if one
    /// was specified in the original wasm module.
    pub fn func_name(&self, idx: FuncIndex) -> Option<&str> {
        // Find entry for `idx`, if present.
        let i = self.func_names.binary_search_by_key(&idx, |n| n.idx).ok()?;
        let name = &self.func_names[i];

        // Here we `unwrap` the `from_utf8` but this can theoretically be a
        // `from_utf8_unchecked` if we really wanted since this section is
        // guaranteed to only have valid utf-8 data. Until it's a problem it's
        // probably best to double-check this though.
        let data = self.code_memory().func_name_data();
        Some(str::from_utf8(&data[name.offset as usize..][..name.len as usize]).unwrap())
    }

    /// Return a reference to a mutable module (if possible).
    pub fn module_mut(&mut self) -> Option<&mut Module> {
        Arc::get_mut(&mut self.module)
    }

    /// Returns an iterator over all functions defined within this module with
    /// their index and their body in memory.
    #[inline]
    pub fn finished_functions(
        &self,
    ) -> impl ExactSizeIterator<Item = (DefinedFuncIndex, &[u8])> + '_ {
        self.funcs
            .iter()
            .map(move |(i, _)| (i, self.finished_function(i)))
    }

    /// Returns the body of the function that `index` points to.
    #[inline]
    pub fn finished_function(&self, index: DefinedFuncIndex) -> &[u8] {
        let loc = self.funcs[index].wasm_func_loc;
        &self.text()[loc.start as usize..][..loc.length as usize]
    }

    /// Get the array-to-Wasm trampoline for the function `index` points to.
    ///
    /// If the function `index` points to does not escape, then `None` is
    /// returned.
    ///
    /// These trampolines are used for array callers (e.g. `Func::new`)
    /// calling Wasm callees.
    pub fn array_to_wasm_trampoline(&self, index: DefinedFuncIndex) -> Option<&[u8]> {
        let loc = self.funcs[index].array_to_wasm_trampoline?;
        Some(&self.text()[loc.start as usize..][..loc.length as usize])
    }

    /// Get the native-to-Wasm trampoline for the function `index` points to.
    ///
    /// If the function `index` points to does not escape, then `None` is
    /// returned.
    ///
    /// These trampolines are used for native callers (e.g. `Func::wrap`)
    /// calling Wasm callees.
    #[inline]
    pub fn native_to_wasm_trampoline(&self, index: DefinedFuncIndex) -> Option<&[u8]> {
        let loc = self.funcs[index].native_to_wasm_trampoline?;
        Some(&self.text()[loc.start as usize..][..loc.length as usize])
    }

    /// Get the Wasm-to-native trampoline for the given signature.
    ///
    /// These trampolines are used for filling in
    /// `VMFuncRef::wasm_call` for `Func::wrap`-style host funcrefs
    /// that don't have access to a compiler when created.
    pub fn wasm_to_native_trampoline(&self, signature: SignatureIndex) -> &[u8] {
        let idx = self
            .wasm_to_native_trampolines
            .binary_search_by_key(&signature, |entry| entry.0)
            .expect("should have a Wasm-to-native trampline for all signatures");
        let (_, loc) = self.wasm_to_native_trampolines[idx];
        &self.text()[loc.start as usize..][..loc.length as usize]
    }

    /// Returns the stack map information for all functions defined in this
    /// module.
    ///
    /// The iterator returned iterates over the span of the compiled function in
    /// memory with the stack maps associated with those bytes.
    pub fn stack_maps(&self) -> impl Iterator<Item = (&[u8], &[StackMapInformation])> {
        self.finished_functions().map(|(_, f)| f).zip(
            self.funcs
                .values()
                .map(|f| &f.wasm_func_info.stack_maps[..]),
        )
    }

    /// Lookups a defined function by a program counter value.
    ///
    /// Returns the defined function index and the relative address of
    /// `text_offset` within the function itself.
    pub fn func_by_text_offset(&self, text_offset: usize) -> Option<(DefinedFuncIndex, u32)> {
        let text_offset = u32::try_from(text_offset).unwrap();

        let index = match self.funcs.binary_search_values_by_key(&text_offset, |e| {
            debug_assert!(e.wasm_func_loc.length > 0);
            // Return the inclusive "end" of the function
            e.wasm_func_loc.start + e.wasm_func_loc.length - 1
        }) {
            Ok(k) => {
                // Exact match, pc is at the end of this function
                k
            }
            Err(k) => {
                // Not an exact match, k is where `pc` would be "inserted"
                // Since we key based on the end, function `k` might contain `pc`,
                // so we'll validate on the range check below
                k
            }
        };

        let CompiledFunctionInfo { wasm_func_loc, .. } = self.funcs.get(index)?;
        let start = wasm_func_loc.start;
        let end = wasm_func_loc.start + wasm_func_loc.length;

        if text_offset < start || end < text_offset {
            return None;
        }

        Some((index, text_offset - wasm_func_loc.start))
    }

    /// Gets the function location information for a given function index.
    pub fn func_loc(&self, index: DefinedFuncIndex) -> &FunctionLoc {
        &self
            .funcs
            .get(index)
            .expect("defined function should be present")
            .wasm_func_loc
    }

    /// Gets the function information for a given function index.
    pub fn wasm_func_info(&self, index: DefinedFuncIndex) -> &WasmFunctionInfo {
        &self
            .funcs
            .get(index)
            .expect("defined function should be present")
            .wasm_func_info
    }

    /// Creates a new symbolication context which can be used to further
    /// symbolicate stack traces.
    ///
    /// Basically this makes a thing which parses debuginfo and can tell you
    /// what filename and line number a wasm pc comes from.
    #[cfg(feature = "addr2line")]
    pub fn symbolize_context(&self) -> Result<Option<SymbolizeContext<'_>>> {
        use gimli::EndianSlice;
        if !self.meta.has_wasm_debuginfo {
            return Ok(None);
        }
        let dwarf = gimli::Dwarf::load(|id| -> Result<_> {
            // Lookup the `id` in the `dwarf` array prepared for this module
            // during module serialization where it's sorted by the `id` key. If
            // found this is a range within the general module's concatenated
            // dwarf section which is extracted here, otherwise it's just an
            // empty list to represent that it's not present.
            let data = self
                .meta
                .dwarf
                .binary_search_by_key(&(id as u8), |(id, _)| *id)
                .map(|i| {
                    let (_, range) = &self.meta.dwarf[i];
                    &self.code_memory().dwarf()[range.start as usize..range.end as usize]
                })
                .unwrap_or(&[]);
            Ok(EndianSlice::new(data, gimli::LittleEndian))
        })?;
        let cx = addr2line::Context::from_dwarf(dwarf)
            .context("failed to create addr2line dwarf mapping context")?;
        Ok(Some(SymbolizeContext {
            inner: cx,
            code_section_offset: self.meta.code_section_offset,
        }))
    }

    /// Returns whether the original wasm module had unparsed debug information
    /// based on the tunables configuration.
    pub fn has_unparsed_debuginfo(&self) -> bool {
        self.meta.has_unparsed_debuginfo
    }

    /// Indicates whether this module came with n address map such that lookups
    /// via `wasmtime_environ::lookup_file_pos` will succeed.
    ///
    /// If this function returns `false` then `lookup_file_pos` will always
    /// return `None`.
    pub fn has_address_map(&self) -> bool {
        !self.code_memory.address_map_data().is_empty()
    }

    /// Returns the bounds, in host memory, of where this module's compiled
    /// image resides.
    pub fn image_range(&self) -> Range<usize> {
        let base = self.mmap().as_ptr() as usize;
        let len = self.mmap().len();
        base..base + len
    }
}

#[cfg(feature = "addr2line")]
type Addr2LineContext<'a> = addr2line::Context<gimli::EndianSlice<'a, gimli::LittleEndian>>;

/// A context which contains dwarf debug information to translate program
/// counters back to filenames and line numbers.
#[cfg(feature = "addr2line")]
pub struct SymbolizeContext<'a> {
    inner: Addr2LineContext<'a>,
    code_section_offset: u64,
}

#[cfg(feature = "addr2line")]
impl<'a> SymbolizeContext<'a> {
    /// Returns access to the [`addr2line::Context`] which can be used to query
    /// frame information with.
    pub fn addr2line(&self) -> &Addr2LineContext<'a> {
        &self.inner
    }

    /// Returns the offset of the code section in the original wasm file, used
    /// to calculate lookup values into the DWARF.
    pub fn code_section_offset(&self) -> u64 {
        self.code_section_offset
    }
}
#[cfg(test)]
mod tests {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
    };

    use crate::{Config, Engine, Module, ModuleVersionStrategy, OptLevel};

    use anyhow::Result;
    use tempfile::TempDir;

    #[test]
    #[cfg_attr(miri, ignore)]
    fn cache_accounts_for_opt_level() -> Result<()> {
        let td = TempDir::new()?;
        let config_path = td.path().join("config.toml");
        std::fs::write(
            &config_path,
            &format!(
                "
                    [cache]
                    enabled = true
                    directory = '{}'
                ",
                td.path().join("cache").display()
            ),
        )?;
        let mut cfg = Config::new();
        cfg.cranelift_opt_level(OptLevel::None)
            .cache_config_load(&config_path)?;
        let engine = Engine::new(&cfg)?;
        Module::new(&engine, "(module (func))")?;
        assert_eq!(engine.config().cache_config.cache_hits(), 0);
        assert_eq!(engine.config().cache_config.cache_misses(), 1);
        Module::new(&engine, "(module (func))")?;
        assert_eq!(engine.config().cache_config.cache_hits(), 1);
        assert_eq!(engine.config().cache_config.cache_misses(), 1);

        let mut cfg = Config::new();
        cfg.cranelift_opt_level(OptLevel::Speed)
            .cache_config_load(&config_path)?;
        let engine = Engine::new(&cfg)?;
        Module::new(&engine, "(module (func))")?;
        assert_eq!(engine.config().cache_config.cache_hits(), 0);
        assert_eq!(engine.config().cache_config.cache_misses(), 1);
        Module::new(&engine, "(module (func))")?;
        assert_eq!(engine.config().cache_config.cache_hits(), 1);
        assert_eq!(engine.config().cache_config.cache_misses(), 1);

        let mut cfg = Config::new();
        cfg.cranelift_opt_level(OptLevel::SpeedAndSize)
            .cache_config_load(&config_path)?;
        let engine = Engine::new(&cfg)?;
        Module::new(&engine, "(module (func))")?;
        assert_eq!(engine.config().cache_config.cache_hits(), 0);
        assert_eq!(engine.config().cache_config.cache_misses(), 1);
        Module::new(&engine, "(module (func))")?;
        assert_eq!(engine.config().cache_config.cache_hits(), 1);
        assert_eq!(engine.config().cache_config.cache_misses(), 1);

        let mut cfg = Config::new();
        cfg.debug_info(true).cache_config_load(&config_path)?;
        let engine = Engine::new(&cfg)?;
        Module::new(&engine, "(module (func))")?;
        assert_eq!(engine.config().cache_config.cache_hits(), 0);
        assert_eq!(engine.config().cache_config.cache_misses(), 1);
        Module::new(&engine, "(module (func))")?;
        assert_eq!(engine.config().cache_config.cache_hits(), 1);
        assert_eq!(engine.config().cache_config.cache_misses(), 1);

        Ok(())
    }

    #[test]
    fn precompile_compatibility_key_accounts_for_opt_level() {
        fn hash_for_config(cfg: &Config) -> u64 {
            let engine = Engine::new(cfg).expect("Config should be valid");
            let mut hasher = DefaultHasher::new();
            engine.precompile_compatibility_hash().hash(&mut hasher);
            hasher.finish()
        }
        let mut cfg = Config::new();
        cfg.cranelift_opt_level(OptLevel::None);
        let opt_none_hash = hash_for_config(&cfg);
        cfg.cranelift_opt_level(OptLevel::Speed);
        let opt_speed_hash = hash_for_config(&cfg);
        assert_ne!(opt_none_hash, opt_speed_hash)
    }

    #[test]
    fn precompile_compatibility_key_accounts_for_module_version_strategy() -> Result<()> {
        fn hash_for_config(cfg: &Config) -> u64 {
            let engine = Engine::new(cfg).expect("Config should be valid");
            let mut hasher = DefaultHasher::new();
            engine.precompile_compatibility_hash().hash(&mut hasher);
            hasher.finish()
        }
        let mut cfg_custom_version = Config::new();
        cfg_custom_version.module_version(ModuleVersionStrategy::Custom("1.0.1111".to_string()))?;
        let custom_version_hash = hash_for_config(&cfg_custom_version);

        let mut cfg_default_version = Config::new();
        cfg_default_version.module_version(ModuleVersionStrategy::WasmtimeVersion)?;
        let default_version_hash = hash_for_config(&cfg_default_version);

        let mut cfg_none_version = Config::new();
        cfg_none_version.module_version(ModuleVersionStrategy::None)?;
        let none_version_hash = hash_for_config(&cfg_none_version);

        assert_ne!(custom_version_hash, default_version_hash);
        assert_ne!(custom_version_hash, none_version_hash);
        assert_ne!(default_version_hash, none_version_hash);

        Ok(())
    }
}
