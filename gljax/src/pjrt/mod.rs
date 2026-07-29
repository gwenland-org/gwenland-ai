//! Safe wrappers over the PJRT C API.
//!
//! Nothing outside this module calls into [`crate::sys`]. The ownership story
//! is a chain of borrows, each link matching a real PJRT lifetime rule:
//!
//! ```text
//! PjrtPlugin  (owns the Library — unloading it dangles everything below)
//!   └─ PjrtClientHandle<'p>
//!        ├─ LoadedExecutable<'c, 'p>
//!        └─ PjrtBufferHandle<'c, 'p>
//! ```

pub mod buffer;
pub mod client;
pub mod compile;
pub mod device;
pub(crate) mod error;
pub(crate) mod event;
pub mod plugin;

pub use buffer::PjrtBufferHandle;
pub use client::PjrtClientHandle;
pub use compile::LoadedExecutable;
pub use device::PjrtDeviceRef;
pub use plugin::{cpu_plugin_path, PjrtPlugin, ENV_PLUGIN_CPU, ENV_PLUGIN_CPU_ALIAS};
