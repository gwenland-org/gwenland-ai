use std::fmt;
use crate::error::{GllmError, GllmResult};

/// Extension URI — identifies a layer type plugin
///
/// Format: `"gllm:transformer/standard@v1"`,
///         `"gllm:transformer/moe@v1"`,
///         `"gllm:mamba/standard@v1"`,
///         `"myorg:custom/fused@v1"`
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExtensionUri {
    pub namespace: String,  // "gllm" atau custom org
    pub category: String,   // "transformer", "mamba", "projector"
    pub name: String,       // "standard", "moe", "mla"
    pub version: String,    // "v1", "v2"
}

impl ExtensionUri {
    pub fn parse(uri: &str) -> GllmResult<Self> {
        // Format: "namespace:category/name@version"
        let (namespace, rest) = uri.split_once(':')
            .ok_or_else(|| GllmError::UnknownExtension(uri.to_string()))?;
        let (cat_name, version) = rest.split_once('@')
            .ok_or_else(|| GllmError::UnknownExtension(uri.to_string()))?;
        let (category, name) = cat_name.split_once('/')
            .ok_or_else(|| GllmError::UnknownExtension(uri.to_string()))?;

        if namespace.is_empty() || category.is_empty() || name.is_empty() || version.is_empty() {
            return Err(GllmError::UnknownExtension(uri.to_string()));
        }

        Ok(Self {
            namespace: namespace.to_string(),
            category: category.to_string(),
            name: name.to_string(),
            version: version.to_string(),
        })
    }

    pub fn is_official(&self) -> bool { self.namespace == "gllm" }
}

impl fmt::Display for ExtensionUri {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}/{}@{}", self.namespace, self.category, self.name, self.version)
    }
}

/// Known built-in layer types
pub mod known_extensions {
    pub const TRANSFORMER_STANDARD: &str = "gllm:transformer/standard@v1";
    pub const TRANSFORMER_MOE: &str = "gllm:transformer/moe@v1";
    pub const TRANSFORMER_MLA: &str = "gllm:transformer/mla@v1";
    pub const MAMBA_STANDARD: &str = "gllm:mamba/standard@v1";
    pub const PROJECTOR_LINEAR: &str = "gllm:projector/linear@v1";
}
