use std::mem::ManuallyDrop;
use std::ops::Deref;

use anyhow::{Context, Result};
use wasmtime_jit::{CodeMemory, LibCalls};
use wasmtime_jit_icache_coherence as icache_coherence;
use wasmtime_runtime::libcalls::relocs;
use wasmtime_runtime::MmapVec;

use crate::unwind::UnwindRegistration;

pub(crate) const LIBCALLS: LibCalls = LibCalls {
    floorf32: relocs::floorf32,
    floorf64: relocs::floorf64,
    nearestf32: relocs::nearestf32,
    nearestf64: relocs::nearestf64,
    ceilf32: relocs::ceilf32,
    ceilf64: relocs::ceilf64,
    truncf32: relocs::truncf32,
    truncf64: relocs::truncf64,
    fmaf32: relocs::fmaf32,
    fmaf64: relocs::fmaf64,
    x86_pshufb: relocs::x86_pshufb,
};

pub struct MmapCodeMemory {
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

impl Deref for MmapCodeMemory {
    type Target = CodeMemory<MmapVec>;

    fn deref(&self) -> &Self::Target {
        &self.memory
    }
}

impl MmapCodeMemory {
    pub fn new(mmap: CodeMemory<MmapVec>) -> Self {
        Self {
            memory: ManuallyDrop::new(mmap),
            unwind_registration: ManuallyDrop::new(None),
            published: false,
        }
    }

    pub fn memory(&self) -> &CodeMemory<MmapVec> {
        &self.memory
    }

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
    pub fn publish(&mut self) -> Result<()> {
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
            self.memory.apply_relocations(LIBCALLS)?;

            // Next freeze the contents of this image by making all of the
            // memory readonly. Nothing after this point should ever be modified
            // so commit everything. For a compiled-in-memory image this will
            // mean IPIs to evict writable mappings from other cores. For
            // loaded-from-disk images this shouldn't result in IPIs so long as
            // there weren't any relocations because nothing should have
            // otherwise written to the image at any point either.
            self.memory
                .mmap()
                .make_readonly(0..self.memory.mmap().len())?;

            let text = self.memory.text();

            // Clear the newly allocated code from cache if the processor requires it
            //
            // Do this before marking the memory as R+X, technically we should be able to do it after
            // but there are some CPU's that have had errata about doing this with read only memory.
            icache_coherence::clear_cache(text.as_ptr().cast(), text.len())
                .expect("Failed cache clear");

            // Switch the executable portion from readonly to read/execute.
            self.memory
                .mmap()
                .make_executable(
                    self.memory.text_range().clone(),
                    self.memory.enable_branch_protection(),
                )
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
