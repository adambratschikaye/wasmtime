use std::str::FromStr;

use anyhow::{anyhow, bail, Result};
use object::{
    write::{Object, StandardSegment},
    SectionKind,
};
use serde::{Deserialize, Serialize};
use wasmtime_environ::{obj, Compiler, FlagValue, Tunables};

use crate::config::ModuleVersionStrategy;

const VERSION: u8 = 0;

#[derive(Serialize, Deserialize)]
pub struct Metadata<'a> {
    target: String,
    #[serde(borrow)]
    shared_flags: Vec<(&'a str, FlagValue<'a>)>,
    #[serde(borrow)]
    isa_flags: Vec<(&'a str, FlagValue<'a>)>,
    tunables: Tunables,
    features: WasmFeatures,
}

// This exists because `wasmparser::WasmFeatures` isn't serializable
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
struct WasmFeatures {
    reference_types: bool,
    multi_value: bool,
    bulk_memory: bool,
    component_model: bool,
    simd: bool,
    tail_call: bool,
    threads: bool,
    multi_memory: bool,
    exceptions: bool,
    memory64: bool,
    relaxed_simd: bool,
    extended_const: bool,
    function_references: bool,
}

impl Metadata<'_> {
    #[cfg(any(feature = "cranelift", feature = "winch"))]
    pub fn new(
        compiler: &dyn wasmtime_environ::Compiler,
        features: wasmparser::WasmFeatures,
        tunables: &Tunables,
    ) -> Metadata<'static> {
        let wasmparser::WasmFeatures {
            reference_types,
            multi_value,
            bulk_memory,
            component_model,
            simd,
            threads,
            tail_call,
            multi_memory,
            exceptions,
            memory64,
            relaxed_simd,
            extended_const,
            memory_control,
            function_references,
            gc,
            component_model_values,

            // Always on; we don't currently have knobs for these.
            mutable_global: _,
            saturating_float_to_int: _,
            sign_extension: _,
            floats: _,
        } = features;

        assert!(!memory_control);
        assert!(!gc);
        assert!(!component_model_values);

        Metadata {
            target: compiler.triple().to_string(),
            shared_flags: compiler.flags(),
            isa_flags: compiler.isa_flags(),
            tunables: tunables.clone(),
            features: WasmFeatures {
                reference_types,
                multi_value,
                bulk_memory,
                component_model,
                simd,
                threads,
                tail_call,
                multi_memory,
                exceptions,
                memory64,
                relaxed_simd,
                extended_const,
                function_references,
            },
        }
    }

    pub fn shared_flags(&self) -> impl Iterator<Item = &(&str, FlagValue<'_>)> {
        self.shared_flags.iter()
    }

    pub fn isa_flags(&self) -> impl Iterator<Item = &(&str, FlagValue<'_>)> {
        self.isa_flags.iter()
    }

    pub fn check_compatible(
        &self,
        compiler: &dyn Compiler,
        tunables: &Tunables,
        features: &wasmparser::WasmFeatures,
    ) -> Result<()> {
        self.check_triple(compiler)?;
        self.check_tunables(tunables)?;
        self.check_features(features)?;
        Ok(())
    }

    fn check_triple(&self, compiler: &dyn Compiler) -> Result<()> {
        let engine_target = compiler.triple();
        let module_target =
            target_lexicon::Triple::from_str(&self.target).map_err(|e| anyhow!(e))?;

        if module_target.architecture != engine_target.architecture {
            bail!(
                "Module was compiled for architecture '{}'",
                module_target.architecture
            );
        }

        if module_target.operating_system != engine_target.operating_system {
            bail!(
                "Module was compiled for operating system '{}'",
                module_target.operating_system
            );
        }

        Ok(())
    }

    fn check_int<T: Eq + std::fmt::Display>(found: T, expected: T, feature: &str) -> Result<()> {
        if found == expected {
            return Ok(());
        }

        bail!(
            "Module was compiled with a {} of '{}' but '{}' is expected for the host",
            feature,
            found,
            expected
        );
    }

    fn check_bool(found: bool, expected: bool, feature: &str) -> Result<()> {
        if found == expected {
            return Ok(());
        }

        bail!(
            "Module was compiled {} {} but it {} enabled for the host",
            if found { "with" } else { "without" },
            feature,
            if expected { "is" } else { "is not" }
        );
    }

    fn check_tunables(&self, other: &Tunables) -> Result<()> {
        let Tunables {
            static_memory_bound,
            static_memory_offset_guard_size,
            dynamic_memory_offset_guard_size,
            generate_native_debuginfo,
            parse_wasm_debuginfo,
            consume_fuel,
            epoch_interruption,
            static_memory_bound_is_maximum,
            guard_before_linear_memory,
            relaxed_simd_deterministic,
            tail_callable,

            // This doesn't affect compilation, it's just a runtime setting.
            dynamic_memory_growth_reserve: _,

            // This does technically affect compilation but modules with/without
            // trap information can be loaded into engines with the opposite
            // setting just fine (it's just a section in the compiled file and
            // whether it's present or not)
            generate_address_map: _,

            // Just a debugging aid, doesn't affect functionality at all.
            debug_adapter_modules: _,
        } = self.tunables;

        Self::check_int(
            static_memory_bound,
            other.static_memory_bound,
            "static memory bound",
        )?;
        Self::check_int(
            static_memory_offset_guard_size,
            other.static_memory_offset_guard_size,
            "static memory guard size",
        )?;
        Self::check_int(
            dynamic_memory_offset_guard_size,
            other.dynamic_memory_offset_guard_size,
            "dynamic memory guard size",
        )?;
        Self::check_bool(
            generate_native_debuginfo,
            other.generate_native_debuginfo,
            "debug information support",
        )?;
        Self::check_bool(
            parse_wasm_debuginfo,
            other.parse_wasm_debuginfo,
            "WebAssembly backtrace support",
        )?;
        Self::check_bool(consume_fuel, other.consume_fuel, "fuel support")?;
        Self::check_bool(
            epoch_interruption,
            other.epoch_interruption,
            "epoch interruption",
        )?;
        Self::check_bool(
            static_memory_bound_is_maximum,
            other.static_memory_bound_is_maximum,
            "pooling allocation support",
        )?;
        Self::check_bool(
            guard_before_linear_memory,
            other.guard_before_linear_memory,
            "guard before linear memory",
        )?;
        Self::check_bool(
            relaxed_simd_deterministic,
            other.relaxed_simd_deterministic,
            "relaxed simd deterministic semantics",
        )?;
        Self::check_bool(tail_callable, other.tail_callable, "WebAssembly tail calls")?;

        Ok(())
    }

    fn check_features(&self, other: &wasmparser::WasmFeatures) -> Result<()> {
        let WasmFeatures {
            reference_types,
            multi_value,
            bulk_memory,
            component_model,
            simd,
            tail_call,
            threads,
            multi_memory,
            exceptions,
            memory64,
            relaxed_simd,
            extended_const,
            function_references,
        } = self.features;

        Self::check_bool(
            reference_types,
            other.reference_types,
            "WebAssembly reference types support",
        )?;
        Self::check_bool(
            multi_value,
            other.multi_value,
            "WebAssembly multi-value support",
        )?;
        Self::check_bool(
            bulk_memory,
            other.bulk_memory,
            "WebAssembly bulk memory support",
        )?;
        Self::check_bool(
            component_model,
            other.component_model,
            "WebAssembly component model support",
        )?;
        Self::check_bool(simd, other.simd, "WebAssembly SIMD support")?;
        Self::check_bool(tail_call, other.tail_call, "WebAssembly tail calls support")?;
        Self::check_bool(threads, other.threads, "WebAssembly threads support")?;
        Self::check_bool(
            multi_memory,
            other.multi_memory,
            "WebAssembly multi-memory support",
        )?;
        Self::check_bool(
            exceptions,
            other.exceptions,
            "WebAssembly exceptions support",
        )?;
        Self::check_bool(
            memory64,
            other.memory64,
            "WebAssembly 64-bit memory support",
        )?;
        Self::check_bool(
            extended_const,
            other.extended_const,
            "WebAssembly extended-const support",
        )?;
        Self::check_bool(
            relaxed_simd,
            other.relaxed_simd,
            "WebAssembly relaxed-simd support",
        )?;
        Self::check_bool(
            function_references,
            other.function_references,
            "WebAssembly function-references support",
        )?;

        Ok(())
    }

    pub fn tunables(&mut self) -> &mut Tunables {
        &mut self.tunables
    }

    pub fn threads_feature(&mut self) -> &mut bool {
        &mut self.features.threads
    }

    pub fn target(&mut self) -> &mut String {
        &mut self.target
    }

    pub fn push_shared_flag(&mut self, key: &'static str, value: FlagValue<'static>) {
        self.shared_flags.push((key, value))
    }

    pub fn push_isa_flag(&mut self, key: &'static str, value: FlagValue<'static>) {
        self.isa_flags.push((key, value))
    }
}

pub fn append_compiler_info(
    compiler: &dyn Compiler,
    features: wasmparser::WasmFeatures,
    tunables: &Tunables,
    module_version: &ModuleVersionStrategy,
    obj: &mut Object<'_>,
) {
    let section = obj.add_section(
        obj.segment_name(StandardSegment::Data).to_vec(),
        obj::ELF_WASM_ENGINE.as_bytes().to_vec(),
        SectionKind::ReadOnlyData,
    );
    let mut data = Vec::new();
    data.push(VERSION);
    let version = match module_version {
        ModuleVersionStrategy::WasmtimeVersion => env!("CARGO_PKG_VERSION"),
        ModuleVersionStrategy::Custom(c) => c,
        ModuleVersionStrategy::None => "",
    };
    // This precondition is checked in Config::module_version:
    assert!(
        version.len() < 256,
        "package version must be less than 256 bytes"
    );
    data.push(version.len() as u8);
    data.extend_from_slice(version.as_bytes());
    bincode::serialize_into(&mut data, &Metadata::new(compiler, features, tunables)).unwrap();
    obj.set_section_data(section, data, 1);
}

pub fn append_bti(compiler: &dyn Compiler, obj: &mut Object<'_>) {
    let section = obj.add_section(
        obj.segment_name(StandardSegment::Data).to_vec(),
        obj::ELF_WASM_BTI.as_bytes().to_vec(),
        SectionKind::ReadOnlyData,
    );
    let contents = if compiler.is_branch_protection_enabled() {
        1
    } else {
        0
    };
    obj.append_section_data(section, &[contents], 1);
}
