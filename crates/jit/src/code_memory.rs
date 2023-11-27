//! Memory management for executable code.

use crate::subslice_range;
use anyhow::{bail, Context, Result};
use object::read::{File, Object, ObjectSection};
use object::ObjectSymbol;
use std::ops::{Deref, DerefMut, Range};
use wasmtime_environ::obj;

/// TODO: doc
pub struct LibCalls {
    /// TODO: doc
    pub floorf32: usize,
    /// TODO: doc
    pub floorf64: usize,
    /// TODO: doc
    pub nearestf32: usize,
    /// TODO: doc
    pub nearestf64: usize,
    /// TODO: doc
    pub ceilf32: usize,
    /// TODO: doc
    pub ceilf64: usize,
    /// TODO: doc
    pub truncf32: usize,
    /// TODO: doc
    pub truncf64: usize,
    /// TODO: doc
    pub fmaf32: usize,
    /// TODO: doc
    pub fmaf64: usize,
    #[cfg(target_arch = "x86_64")]
    /// TODO: doc
    pub x86_pshufb: usize,
}

/// Management of executable memory within a `MmapVec`
///
/// This type consumes ownership of a region of memory and will manage the
/// executable permissions of the contained JIT code as necessary.
pub struct CodeMemory<T> {
    // NB: these are `ManuallyDrop` because `unwind_registration` must be
    // dropped first since it refers to memory owned by `mmap`.
    mmap: T,
    // enable_branch_protection: bool,
    relocations: Vec<(usize, obj::LibCall)>,

    // Ranges within `self.mmap` of where the particular sections lie.
    text: Range<usize>,
    unwind: Range<usize>,
    trap_data: Range<usize>,
    wasm_data: Range<usize>,
    address_map_data: Range<usize>,
    func_name_data: Range<usize>,
    info_data: Range<usize>,
    dwarf: Range<usize>,
}

fn _assert() {
    fn _assert_send_sync<T: Send + Sync>() {}
    _assert_send_sync::<CodeMemory<Vec<u8>>>();
}

impl<T: Deref<Target = [u8]> + DerefMut> CodeMemory<T> {
    /// Creates a new `CodeMemory` by taking ownership of the provided
    /// `MmapVec`.
    ///
    /// The returned `CodeMemory` manages the internal `MmapVec` and the
    /// `publish` method is used to actually make the memory executable.
    pub fn new(mmap: T, unwind_section_name: &str) -> Result<Self> {
        let obj = File::parse(&mmap[..])
            .with_context(|| "failed to parse internal compilation artifact")?;

        let mut relocations = Vec::new();
        let mut text = 0..0;
        let mut unwind = 0..0;
        // let mut enable_branch_protection = None;
        let mut trap_data = 0..0;
        let mut wasm_data = 0..0;
        let mut address_map_data = 0..0;
        let mut func_name_data = 0..0;
        let mut info_data = 0..0;
        let mut dwarf = 0..0;
        for section in obj.sections() {
            let data = section.data()?;
            let name = section.name()?;
            let range = subslice_range(data, &mmap);

            // Double-check that sections are all aligned properly.
            if section.align() != 0 && data.len() != 0 {
                if (data.as_ptr() as u64 - mmap.as_ptr() as u64) % section.align() != 0 {
                    bail!(
                        "section `{}` isn't aligned to {:#x}",
                        section.name().unwrap_or("ERROR"),
                        section.align()
                    );
                }
            }

            match name {
                obj::ELF_WASM_BTI => match data.len() {
                    1 => {} // enable_branch_protection = Some(data[0] != 0),
                    _ => bail!("invalid `{name}` section"),
                },
                ".text" => {
                    text = range;

                    // The text section might have relocations for things like
                    // libcalls which need to be applied, so handle those here.
                    //
                    // Note that only a small subset of possible relocations are
                    // handled. Only those required by the compiler side of
                    // things are processed.
                    for (offset, reloc) in section.relocations() {
                        assert_eq!(reloc.kind(), object::RelocationKind::Absolute);
                        assert_eq!(reloc.encoding(), object::RelocationEncoding::Generic);
                        assert_eq!(usize::from(reloc.size()), std::mem::size_of::<usize>());
                        assert_eq!(reloc.addend(), 0);
                        let sym = match reloc.target() {
                            object::RelocationTarget::Symbol(id) => id,
                            other => panic!("unknown relocation target {other:?}"),
                        };
                        let sym = obj.symbol_by_index(sym).unwrap().name().unwrap();
                        let libcall = obj::LibCall::from_str(sym)
                            .unwrap_or_else(|| panic!("unknown symbol relocation: {sym}"));

                        let offset = usize::try_from(offset).unwrap();
                        relocations.push((offset, libcall));
                    }
                }
                obj::ELF_WASM_DATA => wasm_data = range,
                obj::ELF_WASMTIME_ADDRMAP => address_map_data = range,
                obj::ELF_WASMTIME_TRAPS => trap_data = range,
                obj::ELF_NAME_DATA => func_name_data = range,
                obj::ELF_WASMTIME_INFO => info_data = range,
                obj::ELF_WASMTIME_DWARF => dwarf = range,
                other if other == unwind_section_name => unwind = range,

                _ => log::debug!("ignoring section {name}"),
            }
        }
        Ok(Self {
            mmap: mmap,
            // enable_branch_protection: enable_branch_protection
            //     .ok_or_else(|| anyhow!("missing `{}` section", obj::ELF_WASM_BTI))?,
            text,
            unwind,
            trap_data,
            address_map_data,
            func_name_data,
            dwarf,
            info_data,
            wasm_data,
            relocations,
        })
    }

    /// Returns a reference to the underlying `MmapVec` this memory owns.
    #[inline]
    pub fn mmap(&self) -> &T {
        &self.mmap
    }

    /// Returns the contents of the text section of the ELF executable this
    /// represents.
    #[inline]
    pub fn text(&self) -> &[u8] {
        &self.mmap[self.text.clone()]
    }

    /// Returns the contents of the `ELF_WASMTIME_DWARF` section.
    #[inline]
    pub fn dwarf(&self) -> &[u8] {
        &self.mmap[self.dwarf.clone()]
    }

    /// Returns the data in the `ELF_NAME_DATA` section.
    #[inline]
    pub fn func_name_data(&self) -> &[u8] {
        &self.mmap[self.func_name_data.clone()]
    }

    /// Returns the concatenated list of all data associated with this wasm
    /// module.
    ///
    /// This is used for initialization of memories and all data ranges stored
    /// in a `Module` are relative to the slice returned here.
    #[inline]
    pub fn wasm_data(&self) -> &[u8] {
        &self.mmap[self.wasm_data.clone()]
    }

    /// Returns the encoded address map section used to pass to
    /// `wasmtime_environ::lookup_file_pos`.
    #[inline]
    pub fn address_map_data(&self) -> &[u8] {
        &self.mmap[self.address_map_data.clone()]
    }

    /// Returns the contents of the `ELF_WASMTIME_INFO` section, or an empty
    /// slice if it wasn't found.
    #[inline]
    pub fn wasmtime_info(&self) -> &[u8] {
        &self.mmap[self.info_data.clone()]
    }

    /// Returns the contents of the `ELF_WASMTIME_TRAPS` section, or an empty
    /// slice if it wasn't found.
    #[inline]
    pub fn trap_data(&self) -> &[u8] {
        &self.mmap[self.trap_data.clone()]
    }

    /// TODO: document
    #[inline]
    pub fn unwind(&self) -> &[u8] {
        &self.mmap[self.unwind.clone()]
    }

    /// TODO: document
    pub unsafe fn apply_relocations(&mut self, libcalls: LibCalls) -> Result<()> {
        if self.relocations.is_empty() {
            return Ok(());
        }

        for (offset, libcall) in self.relocations.iter() {
            let offset = self.text.start + offset;
            let libcall = match libcall {
                obj::LibCall::FloorF32 => libcalls.floorf32,
                obj::LibCall::FloorF64 => libcalls.floorf64,
                obj::LibCall::NearestF32 => libcalls.nearestf32,
                obj::LibCall::NearestF64 => libcalls.nearestf64,
                obj::LibCall::CeilF32 => libcalls.ceilf32,
                obj::LibCall::CeilF64 => libcalls.ceilf64,
                obj::LibCall::TruncF32 => libcalls.truncf32,
                obj::LibCall::TruncF64 => libcalls.truncf64,
                obj::LibCall::FmaF32 => libcalls.fmaf32,
                obj::LibCall::FmaF64 => libcalls.fmaf64,
                #[cfg(target_arch = "x86_64")]
                obj::LibCall::X86Pshufb => libcalls.x86_pshufb,
                #[cfg(not(target_arch = "x86_64"))]
                obj::LibCall::X86Pshufb => unreachable!(),
            };
            self.mmap
                .as_mut_ptr()
                .add(offset)
                .cast::<usize>()
                .write_unaligned(libcall);
        }
        Ok(())
    }
}
