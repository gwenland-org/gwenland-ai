//! ARTX01 compatibility shim — the canonical manifest types moved to
//! [`crate::manifest`] in ARTX03. The old placeholder types
//! (`Architecture`, `SharedComponent`, `LayerEntry`, `ProjectorEntry`,
//! the u32-based `ModelMetadata`) were superseded and dropped; nothing
//! outside this module used them.

pub use crate::manifest::{
    GllmManifest, LayerManifest, ModelMetadata, ProjectorManifest, SharedManifest,
};
