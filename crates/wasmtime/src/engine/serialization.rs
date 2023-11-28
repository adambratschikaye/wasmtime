//! This module implements serialization and deserialization of `Engine`
//! configuration data which is embedded into compiled artifacts of Wasmtime.
//!
//! The data serialized here is used to double-check that when a module is
//! loaded from one host onto another that it's compatible with the target host.
//! Additionally though this data is the first data read from a precompiled
//! artifact so it's "extra hardened" to provide reasonable-ish error messages
//! for mismatching wasmtime versions. Once something successfully deserializes
//! here it's assumed it's meant for this wasmtime so error messages are in
//! general much worse afterwards.
//!
//! Wasmtime AOT artifacts are ELF files so the data for the engine here is
//! stored into a section of the output file. The structure of this section is:
//!
//! 1. A version byte, currently `VERSION`.
//! 2. A byte indicating how long the next field is.
//! 3. A version string of the length of the previous byte value.
//! 4. A `bincode`-encoded `Metadata` structure.
//!
//! This is hoped to help distinguish easily Wasmtime-based ELF files from
//! other random ELF files, as well as provide better error messages for
//! using wasmtime artifacts across versions.

use crate::{Engine, Precompiled};
use anyhow::{anyhow, bail, Context, Result};
use object::{File, FileFlags, Object as _, ObjectSection};
use wasmtime_compile::{Metadata, ModuleVersionStrategy};
use wasmtime_environ::{obj, ObjectKind};
use wasmtime_runtime::MmapVec;

const VERSION: u8 = 0;

/// Verifies that the serialized engine in `mmap` is compatible with the
/// `engine` provided.
///
/// This function will verify that the `mmap` provided can be deserialized
/// successfully and that the contents are all compatible with the `engine`
/// provided here, notably compatible wasm features are enabled, compatible
/// compiler options, etc. If a mismatch is found and the compilation metadata
/// specified is incompatible then an error is returned.
pub fn check_compatible(engine: &Engine, mmap: &MmapVec, expected: ObjectKind) -> Result<()> {
    // Parse the input `mmap` as an ELF file and see if the header matches the
    // Wasmtime-generated header. This includes a Wasmtime-specific `os_abi` and
    // the `e_flags` field should indicate whether `expected` matches or not.
    //
    // Note that errors generated here could mean that a precompiled module was
    // loaded as a component, or vice versa, both of which aren't supposed to
    // work.
    //
    // Ideally we'd only `File::parse` once and avoid the linear
    // `section_by_name` search here but the general serialization code isn't
    // structured well enough to make this easy and additionally it's not really
    // a perf issue right now so doing that is left for another day's
    // refactoring.
    let obj = File::parse(&mmap[..]).context("failed to parse precompiled artifact as an ELF")?;
    let expected_e_flags = match expected {
        ObjectKind::Module => obj::EF_WASMTIME_MODULE,
        ObjectKind::Component => obj::EF_WASMTIME_COMPONENT,
    };
    match obj.flags() {
        FileFlags::Elf {
            os_abi: obj::ELFOSABI_WASMTIME,
            abi_version: 0,
            e_flags,
        } if e_flags == expected_e_flags => {}
        _ => bail!("incompatible object file format"),
    }

    let data = obj
        .section_by_name(obj::ELF_WASM_ENGINE)
        .ok_or_else(|| anyhow!("failed to find section `{}`", obj::ELF_WASM_ENGINE))?
        .data()?;
    let (first, data) = data
        .split_first()
        .ok_or_else(|| anyhow!("invalid engine section"))?;
    if *first != VERSION {
        bail!("mismatched version in engine section");
    }
    let (len, data) = data
        .split_first()
        .ok_or_else(|| anyhow!("invalid engine section"))?;
    let len = usize::from(*len);
    let (version, data) = if data.len() < len + 1 {
        bail!("engine section too small")
    } else {
        data.split_at(len)
    };

    match &engine.config().module_version {
        ModuleVersionStrategy::WasmtimeVersion => {
            let version = std::str::from_utf8(version)?;
            if version != env!("CARGO_PKG_VERSION") {
                bail!(
                    "Module was compiled with incompatible Wasmtime version '{}'",
                    version
                );
            }
        }
        ModuleVersionStrategy::Custom(v) => {
            let version = std::str::from_utf8(&version)?;
            if version != v {
                bail!(
                    "Module was compiled with incompatible version '{}'",
                    version
                );
            }
        }
        ModuleVersionStrategy::None => { /* ignore the version info, accept all */ }
    }
    engine.check_compatible_with_metadata(&bincode::deserialize::<Metadata<'_>>(data)?)
}

fn detect_precompiled<'data, R: object::ReadRef<'data>>(
    obj: File<'data, R>,
) -> Option<Precompiled> {
    match obj.flags() {
        FileFlags::Elf {
            os_abi: obj::ELFOSABI_WASMTIME,
            abi_version: 0,
            e_flags: obj::EF_WASMTIME_MODULE,
        } => Some(Precompiled::Module),
        FileFlags::Elf {
            os_abi: obj::ELFOSABI_WASMTIME,
            abi_version: 0,
            e_flags: obj::EF_WASMTIME_COMPONENT,
        } => Some(Precompiled::Component),
        _ => None,
    }
}

pub fn detect_precompiled_bytes(bytes: &[u8]) -> Option<Precompiled> {
    detect_precompiled(File::parse(bytes).ok()?)
}

pub fn detect_precompiled_file(path: impl AsRef<std::path::Path>) -> Result<Option<Precompiled>> {
    let read_cache = object::ReadCache::new(std::fs::File::open(path)?);
    let obj = File::parse(&read_cache)?;
    Ok(detect_precompiled(obj))
}

#[cfg(test)]
mod test {
    use wasmtime_environ::FlagValue;

    use super::*;
    use crate::Config;

    #[test]
    fn test_architecture_mismatch() -> Result<()> {
        let engine = Engine::default();
        let mut metadata = engine.metadata();
        *metadata.target() = "unknown-generic-linux".to_string();

        match engine.check_compatible_with_metadata(&metadata) {
            Ok(_) => unreachable!(),
            Err(e) => assert_eq!(
                e.to_string(),
                "Module was compiled for architecture 'unknown'",
            ),
        }

        Ok(())
    }

    #[test]
    fn test_os_mismatch() -> Result<()> {
        let engine = Engine::default();
        let mut metadata = engine.metadata();

        *metadata.target() = format!(
            "{}-generic-unknown",
            target_lexicon::Triple::host().architecture
        );

        match engine.check_compatible_with_metadata(&metadata) {
            Ok(_) => unreachable!(),
            Err(e) => assert_eq!(
                e.to_string(),
                "Module was compiled for operating system 'unknown'",
            ),
        }

        Ok(())
    }

    #[test]
    fn test_cranelift_flags_mismatch() -> Result<()> {
        let engine = Engine::default();
        let mut metadata = engine.metadata();

        metadata.push_shared_flag("preserve_frame_pointers", FlagValue::Bool(false));

        match engine.check_compatible_with_metadata(&metadata) {
            Ok(_) => unreachable!(),
            Err(e) => assert!(format!("{:?}", e).starts_with(
                "\
compilation settings of module incompatible with native host

Caused by:
    setting \"preserve_frame_pointers\" is configured to Bool(false) which is not supported"
            )),
        }

        Ok(())
    }

    #[test]
    fn test_isa_flags_mismatch() -> Result<()> {
        let engine = Engine::default();
        let mut metadata = engine.metadata();

        metadata.push_isa_flag("not_a_flag", FlagValue::Bool(true));

        match engine.check_compatible_with_metadata(&metadata) {
            Ok(_) => unreachable!(),
            Err(e) => assert!(format!("{:?}", e).starts_with(
                "\
compilation settings of module incompatible with native host

Caused by:
    cannot test if target-specific flag \"not_a_flag\" is available at runtime",
            )),
        }

        Ok(())
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_tunables_int_mismatch() -> Result<()> {
        let engine = Engine::default();
        let mut metadata = engine.metadata();

        metadata.tunables().static_memory_offset_guard_size = 0;

        match engine.check_compatible_with_metadata(&metadata) {
            Ok(_) => unreachable!(),
            Err(e) => assert_eq!(e.to_string(), "Module was compiled with a static memory guard size of '0' but '2147483648' is expected for the host"),
        }

        Ok(())
    }

    #[test]
    fn test_tunables_bool_mismatch() -> Result<()> {
        let mut config = Config::new();
        config.epoch_interruption(true);

        let engine = Engine::new(&config)?;
        let mut metadata = engine.metadata();
        metadata.tunables().epoch_interruption = false;

        match engine.check_compatible_with_metadata(&metadata) {
            Ok(_) => unreachable!(),
            Err(e) => assert_eq!(
                e.to_string(),
                "Module was compiled without epoch interruption but it is enabled for the host"
            ),
        }

        let mut config = Config::new();
        config.epoch_interruption(false);

        let engine = Engine::new(&config)?;
        let mut metadata = engine.metadata();
        metadata.tunables().epoch_interruption = true;

        match engine.check_compatible_with_metadata(&metadata) {
            Ok(_) => unreachable!(),
            Err(e) => assert_eq!(
                e.to_string(),
                "Module was compiled with epoch interruption but it is not enabled for the host"
            ),
        }

        Ok(())
    }

    #[test]
    fn test_feature_mismatch() -> Result<()> {
        let mut config = Config::new();
        config.wasm_threads(true);

        let engine = Engine::new(&config)?;
        let mut metadata = engine.metadata();
        *metadata.threads_feature() = false;

        match engine.check_compatible_with_metadata(&metadata) {
            Ok(_) => unreachable!(),
            Err(e) => assert_eq!(e.to_string(), "Module was compiled without WebAssembly threads support but it is enabled for the host"),
        }

        let mut config = Config::new();
        config.wasm_threads(false);

        let engine = Engine::new(&config)?;
        let mut metadata = engine.metadata();
        *metadata.threads_feature() = true;

        match engine.check_compatible_with_metadata(&metadata) {
            Ok(_) => unreachable!(),
            Err(e) => assert_eq!(e.to_string(), "Module was compiled with WebAssembly threads support but it is not enabled for the host"),
        }

        Ok(())
    }
}
