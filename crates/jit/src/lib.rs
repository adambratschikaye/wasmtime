//! JIT-style runtime for WebAssembly using Cranelift.

#![deny(missing_docs, trivial_numeric_casts, unused_extern_crates)]
#![warn(unused_import_braces)]

mod code_memory;
mod debug;
mod demangling;
mod instantiate;

pub use crate::code_memory::{CodeMemory, LibCalls};
// #[cfg(feature = "addr2line")]
// pub use crate::instantiate::SymbolizeContext;
pub use crate::instantiate::{
    subslice_range, CompiledFunctionInfo, CompiledModuleInfo, FinishedObject, FunctionName,
    Metadata, ObjectBuilder,
};
pub use debug::create_gdbjit_image;
pub use demangling::*;

/// Version number of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
