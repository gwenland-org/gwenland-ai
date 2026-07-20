//! Layer-type extension registry (ARTX8).
//!
//! A manifest names each layer's type by URI; the registry maps that URI to a
//! [`LayerPlugin`] that knows the type's tensor layout. Resolution happens
//! once per layer at load time, so a package can be checked structurally
//! before any execution begins.
//!
//! ## In-process registration only
//!
//! ARTX8 §"Plugin Loading" describes `dlopen`-ing shared libraries from a
//! plugin path and calling an `extern "C"` registration hook. That is not
//! implemented here, deliberately:
//!
//! - it requires C bindings, which `inference-first.md` rule 6 rules out; and
//! - it loads arbitrary native code from a filesystem path into the
//!   inference process.
//!
//! Plugins are therefore registered in Rust, by the embedder. Everything else
//! ARTX8 specifies — URI resolution, exact-version matching, per-type layout
//! validation — works fully without dynamic loading. If out-of-process
//! extensions are ever needed, a sandboxed boundary is the design to revisit,
//! not `dlopen`.

use std::collections::HashMap;

use crate::error::{GllmError, GllmResult};
use crate::manifest::{DType, ExtensionUri, GllmManifest, TensorEntry, known_extensions};
use crate::traits::plugin::LayerPlugin;

/// Maps extension URIs to the plugins that describe them.
///
/// [`with_builtins`](Self::with_builtins) provides the layer types this crate
/// ships; an embedder adds its own with [`register`](Self::register).
#[derive(Default)]
pub struct PluginRegistry {
    plugins: HashMap<String, Box<dyn LayerPlugin>>,
}

impl PluginRegistry {
    /// An empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// A registry preloaded with the built-in layer types: standard
    /// transformer, MoE, MLA, Mamba, and the linear projector.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        // Built-in URIs are distinct by construction, so registration cannot
        // collide here.
        for plugin in builtin_plugins() {
            registry
                .register(plugin)
                .expect("built-in plugin URIs are unique");
        }
        registry
    }

    /// Add a plugin.
    ///
    /// Returns [`GllmError::UnknownExtension`] if another plugin already
    /// claims the same URI. Registration refuses rather than overwrites: two
    /// plugins disagreeing about one layer type is a wiring bug, and silently
    /// keeping the last one registered would make behaviour depend on
    /// registration order.
    pub fn register(&mut self, plugin: Box<dyn LayerPlugin>) -> GllmResult<()> {
        let uri = plugin.uri().to_string();
        if self.plugins.contains_key(&uri) {
            return Err(GllmError::UnknownExtension(format!(
                "{uri}: already registered; a URI maps to exactly one plugin"
            )));
        }
        self.plugins.insert(uri, plugin);
        Ok(())
    }

    /// Look up a plugin by URI.
    ///
    /// Matching is exact, including the `@vN` suffix: ARTX8 §"Plugin
    /// Versioning" requires that a `v2` plugin does not serve `v1` layers.
    pub fn resolve(&self, uri: &ExtensionUri) -> Option<&dyn LayerPlugin> {
        self.plugins.get(&uri.0).map(|b| b.as_ref())
    }

    /// Look up a plugin, erroring when absent.
    pub fn require(&self, uri: &ExtensionUri) -> GllmResult<&dyn LayerPlugin> {
        self.resolve(uri)
            .ok_or_else(|| GllmError::UnknownExtension(uri.0.clone()))
    }

    /// Whether a URI is registered.
    pub fn contains(&self, uri: &ExtensionUri) -> bool {
        self.plugins.contains_key(&uri.0)
    }

    /// Registered URIs, sorted for stable output.
    pub fn registered_uris(&self) -> Vec<&str> {
        let mut uris: Vec<&str> = self.plugins.keys().map(String::as_str).collect();
        uris.sort_unstable();
        uris
    }

    /// Number of registered plugins.
    pub fn len(&self) -> usize {
        self.plugins.len()
    }

    /// Whether nothing is registered.
    pub fn is_empty(&self) -> bool {
        self.plugins.is_empty()
    }

    /// Check that every layer type a manifest declares has a plugin.
    ///
    /// Returns the unresolvable URIs, in manifest order, with duplicates
    /// removed — empty means the package's layer types are all supported.
    /// This is the load-time gate: it runs before any layer is mapped, so an
    /// unsupported package fails immediately rather than at layer 40 of 80.
    pub fn missing_for_manifest(&self, manifest: &GllmManifest) -> Vec<String> {
        let mut missing: Vec<String> = Vec::new();
        let mut seen: Vec<&str> = Vec::new();

        let types = manifest
            .layers
            .iter()
            .map(|l| &l.layer_type)
            .chain(manifest.projector.as_ref().map(|p| &p.projector_type));

        for uri in types {
            if seen.contains(&uri.0.as_str()) {
                continue;
            }
            seen.push(&uri.0);
            if !self.contains(uri) {
                missing.push(uri.0.clone());
            }
        }
        missing
    }
}

impl std::fmt::Debug for PluginRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PluginRegistry")
            .field("registered", &self.registered_uris())
            .finish()
    }
}

/// The layer types this crate ships.
fn builtin_plugins() -> Vec<Box<dyn LayerPlugin>> {
    vec![
        Box::new(TransformerStandardPlugin::new()),
        Box::new(MoePlugin::new()),
        Box::new(MlaPlugin::new()),
        Box::new(MambaPlugin::new()),
        Box::new(LinearProjectorPlugin::new()),
    ]
}

/// Every dtype the format defines. Built-in plugins are layout descriptions,
/// not kernels, so none of them restricts dtype — what a *backend* can
/// actually compute is the backend's own question
/// ([`ExecutionBackend`](crate::runtime::ExecutionBackend)).
fn accepts_any_dtype(_dtype: DType) -> bool {
    true
}

macro_rules! builtin_plugin {
    (
        $(#[$meta:meta])*
        $name:ident, $uri:expr, [$($tensor:expr),* $(,)?]
    ) => {
        $(#[$meta])*
        #[derive(Debug)]
        pub struct $name {
            uri: ExtensionUri,
        }

        impl $name {
            /// Construct the plugin.
            pub fn new() -> Self {
                Self {
                    uri: ExtensionUri::parse($uri).expect("built-in URI is well-formed"),
                }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl LayerPlugin for $name {
            fn uri(&self) -> &ExtensionUri {
                &self.uri
            }

            fn required_tensors(&self) -> &[&'static str] {
                &[$($tensor),*]
            }

            fn supports_dtype(&self, dtype: DType) -> bool {
                accepts_any_dtype(dtype)
            }
        }
    };
}

builtin_plugin!(
    /// Standard transformer block: attention + feed-forward (ARTX4).
    ///
    /// The norm tensors are `attn_norm.weight` and `ffn_norm.weight`, which is
    /// what the ARTX7 converter actually writes (it preserves GGUF's names
    /// after stripping the `blk.N.` prefix). ARTX4's prose calls them
    /// `input_norm` / `post_attn_norm`; those names appear in no real package,
    /// and requiring them rejected all 24 layers of Qwen2.5-0.5B. Verified
    /// against the converted Qwen2.5-0.5B / 1.5B and Qwen3-1.7B packages.
    ///
    /// Biases (`attn_q.bias` and friends) are present in Qwen2 but absent in
    /// many other models, so they are not required — a layer may always carry
    /// more than the type mandates.
    TransformerStandardPlugin,
    known_extensions::TRANSFORMER_STANDARD,
    [
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    ]
);

builtin_plugin!(
    /// Multi-Head Latent Attention (ARTX8 §MLA).
    ///
    /// Keys and values are compressed into a latent of dimension `L < D`, so
    /// there is a single `attn_kv_latent.weight` where a standard block has
    /// separate K and V projections.
    ///
    /// ⚠️ **Unverified against a real model.** These names are transcribed
    /// from ARTX8's table; no MLA model has been converted yet. The standard
    /// transformer's names turned out to differ from ARTX4's prose (see
    /// [`TransformerStandardPlugin`]), so expect the same here and re-check
    /// against a converted DeepSeek-V2-class model before trusting it.
    MlaPlugin,
    known_extensions::TRANSFORMER_MLA,
    [
        "attn_norm.weight",
        "attn_q.weight",
        "attn_kv_latent.weight",
        "attn_o.weight",
        "ffn_norm.weight",
    ]
);

builtin_plugin!(
    /// Mamba state-space layer (ARTX8 §Mamba).
    MambaPlugin,
    known_extensions::MAMBA_STANDARD,
    [
        "in_proj.weight",
        "conv1d.weight",
        "x_proj.weight",
        "dt_proj.weight",
        "A_log",
        "D",
        "out_proj.weight",
    ]
);

builtin_plugin!(
    /// Linear projector for multimodal packages (`GLLMProj.gllm`).
    LinearProjectorPlugin,
    known_extensions::PROJECTOR_LINEAR,
    ["proj.weight"]
);

/// Mixture-of-Experts block (ARTX8 §MoE).
///
/// Only the router gate and the pre-MoE norm are fixed; the expert tensors
/// themselves are counted rather than named, since their number varies per
/// model. [`validate_layout`](LayerPlugin::validate_layout) checks that the
/// experts present form a complete, gap-free set.
///
/// ⚠️ **Unverified against a real model.** No MoE model has been converted
/// yet, and GGUF stores experts *stacked* (`ffn_gate_exps.weight` carrying all
/// experts in one tensor) rather than as the per-expert `expert_N.*` tensors
/// ARTX8 describes. Whether the converter should unstack them, and whether the
/// stacking order is what the runtime assumes, is an open question tracked in
/// the engine work — this plugin describes ARTX8's layout, not GGUF's. Treat a
/// pass here as "matches the spec", not "will route correctly".
#[derive(Debug)]
pub struct MoePlugin {
    uri: ExtensionUri,
}

impl MoePlugin {
    /// Construct the plugin.
    pub fn new() -> Self {
        Self {
            uri: ExtensionUri::parse(known_extensions::TRANSFORMER_MOE)
                .expect("built-in URI is well-formed"),
        }
    }

    /// The three projections every expert must have.
    pub const EXPERT_PROJECTIONS: [&'static str; 3] =
        ["ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"];

    /// Highest expert index present, or `None` when there are no experts.
    ///
    /// Parses `expert_<N>.<projection>` names; anything else is ignored.
    pub fn max_expert_index(index: &[TensorEntry]) -> Option<u32> {
        index.iter().filter_map(|t| Self::expert_index_of(&t.name)).max()
    }

    /// Expert index encoded in a tensor name, if it names an expert tensor.
    fn expert_index_of(name: &str) -> Option<u32> {
        let rest = name.strip_prefix("expert_")?;
        let (digits, projection) = rest.split_once('.')?;
        Self::EXPERT_PROJECTIONS
            .contains(&projection)
            .then(|| digits.parse().ok())
            .flatten()
    }
}

impl Default for MoePlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl LayerPlugin for MoePlugin {
    fn uri(&self) -> &ExtensionUri {
        &self.uri
    }

    fn required_tensors(&self) -> &[&'static str] {
        &["input_norm.weight", "gate.weight"]
    }

    /// Checks the router tensors, then that experts `0..=max` each carry all
    /// three projections.
    ///
    /// A gap or a half-populated expert is reported rather than tolerated: a
    /// missing `expert_5.ffn_up.weight` would otherwise surface as silently
    /// wrong routing rather than a load error. This is a *structural* check —
    /// it says the expert set is complete, not that the weights are the ones
    /// the source model intended.
    fn validate_layout(&self, index: &[TensorEntry]) -> GllmResult<()> {
        let missing: Vec<&str> = self
            .required_tensors()
            .iter()
            .copied()
            .filter(|name| !index.iter().any(|t| t.name == *name))
            .collect();
        if !missing.is_empty() {
            return Err(GllmError::TensorEntryInvalid(format!(
                "{}: missing required tensor(s): {}",
                self.uri,
                missing.join(", ")
            )));
        }

        let Some(max) = Self::max_expert_index(index) else {
            return Err(GllmError::TensorEntryInvalid(format!(
                "{}: no expert_N.* tensors found",
                self.uri
            )));
        };

        let mut incomplete: Vec<String> = Vec::new();
        for expert in 0..=max {
            for projection in Self::EXPERT_PROJECTIONS {
                let name = format!("expert_{expert}.{projection}");
                if !index.iter().any(|t| t.name == name) {
                    incomplete.push(name);
                }
            }
        }

        if incomplete.is_empty() {
            Ok(())
        } else {
            Err(GllmError::TensorEntryInvalid(format!(
                "{}: experts 0..={max} must each have {} projections; missing: {}",
                self.uri,
                Self::EXPERT_PROJECTIONS.len(),
                incomplete.join(", ")
            )))
        }
    }

    fn supports_dtype(&self, dtype: DType) -> bool {
        accepts_any_dtype(dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(name: &str, size: u64) -> TensorEntry {
        TensorEntry {
            name: name.to_string(),
            shape: vec![size],
            dtype: DType::Q8_0,
            offset: 0,
            size,
        }
    }

    fn tensors(names: &[&str]) -> Vec<TensorEntry> {
        names.iter().map(|n| tensor(n, 64)).collect()
    }

    fn uri(s: &str) -> ExtensionUri {
        ExtensionUri::parse(s).unwrap()
    }

    // --- registry ----------------------------------------------------------

    #[test]
    fn builtins_cover_every_known_extension() {
        let reg = PluginRegistry::with_builtins();
        assert_eq!(reg.len(), 5);
        for u in [
            known_extensions::TRANSFORMER_STANDARD,
            known_extensions::TRANSFORMER_MOE,
            known_extensions::TRANSFORMER_MLA,
            known_extensions::MAMBA_STANDARD,
            known_extensions::PROJECTOR_LINEAR,
        ] {
            assert!(reg.contains(&uri(u)), "{u} must be registered");
        }
    }

    #[test]
    fn resolve_matches_the_exact_version() {
        let reg = PluginRegistry::with_builtins();
        assert!(reg.resolve(&uri("gllm:transformer/standard@v1")).is_some());
        // ARTX8 §Plugin Versioning: v2 is a different layer type entirely.
        assert!(reg.resolve(&uri("gllm:transformer/standard@v2")).is_none());
    }

    #[test]
    fn require_names_the_unresolved_uri() {
        let reg = PluginRegistry::with_builtins();
        // `unwrap_err` would require `Debug` on `dyn LayerPlugin`; matching
        // keeps that bound off the trait, which plugin authors would have to
        // satisfy for no benefit.
        match reg.require(&uri("myorg:custom/fused@v1")) {
            Err(GllmError::UnknownExtension(u)) => assert_eq!(u, "myorg:custom/fused@v1"),
            Err(other) => panic!("expected UnknownExtension, got {other:?}"),
            Ok(_) => panic!("an unregistered URI must not resolve"),
        }
    }

    #[test]
    fn registering_a_duplicate_uri_is_refused() {
        let mut reg = PluginRegistry::new();
        reg.register(Box::new(MoePlugin::new())).unwrap();
        let err = reg.register(Box::new(MoePlugin::new())).unwrap_err();
        assert!(
            matches!(err, GllmError::UnknownExtension(_)),
            "a URI must map to exactly one plugin, got {err:?}"
        );
        assert_eq!(reg.len(), 1, "the first registration stands");
    }

    #[test]
    fn a_custom_plugin_can_be_registered() {
        struct Custom(ExtensionUri);
        impl LayerPlugin for Custom {
            fn uri(&self) -> &ExtensionUri {
                &self.0
            }
            fn required_tensors(&self) -> &[&'static str] {
                &["fused.weight"]
            }
            fn supports_dtype(&self, _d: DType) -> bool {
                true
            }
        }

        let mut reg = PluginRegistry::with_builtins();
        reg.register(Box::new(Custom(uri("myorg:custom/fused@v1"))))
            .unwrap();
        assert_eq!(reg.len(), 6);

        let p = reg.require(&uri("myorg:custom/fused@v1")).unwrap();
        p.validate_layout(&tensors(&["fused.weight"])).unwrap();
        assert!(p.validate_layout(&tensors(&["other.weight"])).is_err());
    }

    #[test]
    fn registered_uris_are_sorted() {
        let reg = PluginRegistry::with_builtins();
        let uris = reg.registered_uris();
        let mut sorted = uris.clone();
        sorted.sort_unstable();
        assert_eq!(uris, sorted);
    }

    #[test]
    fn an_empty_registry_reports_itself_empty() {
        let reg = PluginRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
        assert!(!reg.contains(&uri(known_extensions::TRANSFORMER_STANDARD)));
    }

    // --- layout validation -------------------------------------------------

    #[test]
    fn standard_transformer_accepts_a_complete_layer() {
        let p = TransformerStandardPlugin::new();
        p.validate_layout(&tensors(p.required_tensors())).unwrap();
    }

    #[test]
    fn standard_transformer_accepts_a_real_qwen2_layer() {
        // The exact tensor set the ARTX7 converter wrote for layer 0 of
        // Qwen2.5-0.5B, biases included. Transcribing ARTX4's prose names
        // (input_norm / post_attn_norm) instead of these rejected all 24
        // layers of a package that is byte-identical to its source.
        let p = TransformerStandardPlugin::new();
        p.validate_layout(&tensors(&[
            "attn_norm.weight",
            "ffn_down.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_norm.weight",
            "attn_k.bias",
            "attn_k.weight",
            "attn_output.weight",
            "attn_q.bias",
            "attn_q.weight",
            "attn_v.bias",
            "attn_v.weight",
        ]))
        .expect("a real converted Qwen2 layer must validate");
    }

    #[test]
    fn standard_transformer_accepts_a_layer_without_biases() {
        // Qwen3 drops the attention biases Qwen2 carries.
        let p = TransformerStandardPlugin::new();
        p.validate_layout(&tensors(&[
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ]))
        .unwrap();
    }

    #[test]
    fn missing_tensors_are_all_reported_at_once() {
        // One name per run would make a converter bug tedious to chase.
        let p = TransformerStandardPlugin::new();
        let err = p
            .validate_layout(&tensors(&["input_norm.weight", "attn_q.weight"]))
            .unwrap_err();
        let msg = err.to_string();
        for expected in ["attn_k.weight", "attn_v.weight", "ffn_down.weight"] {
            assert!(msg.contains(expected), "{expected} missing from: {msg}");
        }
    }

    #[test]
    fn extra_tensors_are_allowed() {
        // Required means required, not exhaustive — a layer may carry biases
        // or vendor extras the type does not mandate.
        let p = TransformerStandardPlugin::new();
        let mut idx = tensors(p.required_tensors());
        idx.push(tensor("attn_q.bias", 8));
        p.validate_layout(&idx).unwrap();
    }

    #[test]
    fn mla_requires_the_latent_projection_not_separate_kv() {
        let p = MlaPlugin::new();
        p.validate_layout(&tensors(p.required_tensors())).unwrap();

        // A standard block's tensors must not pass as MLA.
        let standard = TransformerStandardPlugin::new();
        let err = p
            .validate_layout(&tensors(standard.required_tensors()))
            .unwrap_err();
        assert!(err.to_string().contains("attn_kv_latent.weight"), "{err}");
    }

    #[test]
    fn mamba_requires_its_state_space_parameters() {
        let p = MambaPlugin::new();
        p.validate_layout(&tensors(p.required_tensors())).unwrap();

        let err = p
            .validate_layout(&tensors(&["in_proj.weight", "out_proj.weight"]))
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("A_log") && msg.contains("conv1d.weight"), "{msg}");
    }

    // --- MoE ---------------------------------------------------------------

    fn moe_tensors(experts: u32) -> Vec<TensorEntry> {
        let mut idx = tensors(&["input_norm.weight", "gate.weight"]);
        for e in 0..experts {
            for p in MoePlugin::EXPERT_PROJECTIONS {
                idx.push(tensor(&format!("expert_{e}.{p}"), 64));
            }
        }
        idx
    }

    #[test]
    fn moe_accepts_a_complete_expert_set() {
        let p = MoePlugin::new();
        p.validate_layout(&moe_tensors(8)).unwrap();
        assert_eq!(MoePlugin::max_expert_index(&moe_tensors(8)), Some(7));
    }

    #[test]
    fn moe_accepts_qwen3_scale_expert_counts() {
        // Qwen3 MoE runs 128 experts; the check must not assume a small set.
        let p = MoePlugin::new();
        p.validate_layout(&moe_tensors(128)).unwrap();
        assert_eq!(MoePlugin::max_expert_index(&moe_tensors(128)), Some(127));
    }

    #[test]
    fn moe_rejects_a_half_populated_expert() {
        // The failure this guards against: a missing projection would surface
        // as silently wrong routing, not as a load error.
        let p = MoePlugin::new();
        let mut idx = moe_tensors(4);
        idx.retain(|t| t.name != "expert_2.ffn_up.weight");

        let err = p.validate_layout(&idx).unwrap_err();
        assert!(
            err.to_string().contains("expert_2.ffn_up.weight"),
            "the incomplete expert must be named: {err}"
        );
    }

    #[test]
    fn moe_rejects_a_gap_in_the_expert_sequence() {
        let p = MoePlugin::new();
        let mut idx = moe_tensors(5);
        idx.retain(|t| !t.name.starts_with("expert_3."));

        let err = p.validate_layout(&idx).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("expert_3."), "the gap must be named: {msg}");
    }

    #[test]
    fn moe_rejects_a_layer_with_no_experts() {
        let p = MoePlugin::new();
        let err = p.validate_layout(&moe_tensors(0)).unwrap_err();
        assert!(err.to_string().contains("no expert_N.* tensors"), "{err}");
    }

    #[test]
    fn moe_rejects_a_layer_missing_its_router() {
        let p = MoePlugin::new();
        let mut idx = moe_tensors(4);
        idx.retain(|t| t.name != "gate.weight");

        let err = p.validate_layout(&idx).unwrap_err();
        assert!(err.to_string().contains("gate.weight"), "{err}");
    }

    #[test]
    fn expert_index_parsing_ignores_unrelated_names() {
        let idx = vec![
            tensor("expert_notanumber.ffn_up.weight", 8),
            tensor("expert_3.unknown_projection", 8),
            tensor("gate.weight", 8),
        ];
        assert_eq!(
            MoePlugin::max_expert_index(&idx),
            None,
            "none of these name a real expert projection"
        );
    }

    // --- memory ------------------------------------------------------------

    #[test]
    fn memory_requirement_sums_the_tensor_index() {
        let p = TransformerStandardPlugin::new();
        let idx = vec![tensor("a", 1024), tensor("b", 2048)];
        assert_eq!(
            p.memory_requirement_bytes(&idx, &crate::types::execution::Device::Cpu),
            3072
        );
    }

    #[test]
    fn builtin_plugins_do_not_restrict_dtype() {
        // Layout descriptions, not kernels — what can actually be computed is
        // the backend's question.
        let p = MoePlugin::new();
        for d in [DType::F32, DType::Q4K, DType::Q8_0, DType::Bf16] {
            assert!(p.supports_dtype(d));
        }
    }
}
