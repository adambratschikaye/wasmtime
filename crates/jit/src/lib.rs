//! JIT-style runtime for WebAssembly using Cranelift.

#![deny(missing_docs, trivial_numeric_casts, unused_extern_crates)]
#![warn(unused_import_braces)]

mod debug;
mod demangling;
mod instantiate;

pub use crate::instantiate::{
    CompiledFunctionInfo, CompiledModuleInfo, FinishedObject, FunctionName, Metadata, ObjectBuilder,
};
pub use debug::create_gdbjit_image;
pub use demangling::*;

/// Version number of this crate.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
