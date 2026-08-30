//! Stummañ Deskiñ: the two output modes — a live line per step, and the
//! report written when the run ends.
//!
//! # JSON is built on the in-tree encoder, not serde
//!
//! `checkpoint/json.rs` already carries a writer and a parser, written for
//! safetensors headers. Reusing it keeps gltrain's dependency list at four
//! crates and means the report is parsed back by the same code that parses a
//! checkpoint header — so the round-trip test in this file exercises a path
//! the checkpoint format depends on too.
//!
//! # `n/a` is printed, never `0`
//!
//! Every platform-dependent field renders as `n/a` when it is `None`. A
//! dashboard that shows `0 °C` because a sensor was missing is worse than one
//! that shows nothing, because the first invites a conclusion.

use super::anomaly::{ENHealthLevel, ENVerdict, VLAnomaly};
use super::axes::VLObservedStep;
use crate::checkpoint::json::{self, Json};
use crate::error::{GlTrainError, Result};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// What the run was, recorded alongside what it did.
///
/// `VL` because it is a plain config bag with derived traits only.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct VLRunConfig {
    /// Free-form model description, e.g. `"ABEmbedding[259,64] + ABLinear[64,259]"`.
    pub model: String,
    /// Where the data came from.
    pub dataset: String,
    /// Sequences per batch.
    pub batch_size: usize,
    /// Learning rate.
    pub lr: f64,
    /// Steps planned.
    pub steps: usize,
    /// A stamp identifying the run. Seconds since the Unix epoch, as a string,
    /// so the report filenames sort chronologically.
    pub timestamp: String,
}

impl VLRunConfig {
    /// The current wall-clock stamp, or `"0"` if the clock is before the epoch.
    pub fn now_stamp() -> String {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs().to_string())
            .unwrap_or_else(|_| "0".to_string())
    }
}

/// Render an `Option<f32>` with a unit, or `n/a`.
fn opt_f32(v: Option<f32>, unit: &str, decimals: usize) -> String {
    match v {
        Some(x) => format!("{x:.*}{unit}", decimals),
        None => "n/a".to_string(),
    }
}

/// A `Json` number that is `null` when the reading was absent.
fn jnum_opt<T: Into<f64>>(v: Option<T>) -> Json {
    v.map_or(Json::Null, |x| Json::n(x.into()))
}

/// A JSON number that stays parseable when the value is not finite.
///
/// JSON has no `NaN` or `Infinity` literal, and emitting one produces a
/// document no parser will read back — including this crate's own. A NaN loss
/// is exactly the case the report exists to record, so it becomes `null` and
/// the `numerical` block says which flag fired.
fn jnum_finite(v: f64) -> Json {
    if v.is_finite() {
        Json::n(v)
    } else {
        Json::Null
    }
}

/// The compact one-line summary printed every step.
///
/// Deliberately fixed-width per field so ten of them scan as a table.
pub fn live_line(step: &VLObservedStep, total_steps: usize) -> String {
    let d = &step.dynamics;
    let t = &step.timing;
    let ppl = if d.perplexity.is_finite() {
        format!("{:.1}", d.perplexity)
    } else {
        "inf".to_string()
    };

    let ram = match step.memory.heap_rss_mb() {
        Some(mb) if mb >= 1024.0 => format!("{:.1}GB", mb / 1024.0),
        Some(mb) => format!("{mb:.0}MB"),
        None => "n/a".to_string(),
    };
    let cpu = format!(
        "{} {}",
        opt_f32(step.hardware.cpu_temp_celsius, "C", 0),
        match step.hardware.cpu_freq_mhz {
            Some(mhz) => format!("{:.1}GHz", mhz / 1000.0),
            None => "n/a".to_string(),
        }
    );

    format!(
        "Step {:>3}/{} | loss {:.4} {}{:.4} ppl {:>7} | g {:.4} dt {:.4} | fwd {:.1}ms bwd {:.1}ms opt {:.1}ms | {:.1} tok/s | RAM {} | CPU {} | {}",
        step.step,
        total_steps,
        d.loss,
        if d.delta_loss <= 0.0 { "" } else { "+" },
        d.delta_loss,
        ppl,
        step.global_grad_norm,
        step.global_update_norm,
        t.forward_ms,
        t.backward_ms,
        t.optimizer_ms,
        t.supervised_tokens_per_sec,
        ram,
        cpu,
        step.health.level.badge()
    )
}

/// The indented anomaly lines printed under a live line.
pub fn anomaly_lines(step: &VLObservedStep) -> Vec<String> {
    step.health
        .anomalies
        .iter()
        .map(|a| format!("  -> [{}/{}] {}", a.axis, a.code, a.message))
        .collect()
}

/// The run's outcome and the sentence explaining it.
pub fn verdict(steps: &[VLObservedStep]) -> (ENVerdict, String) {
    let mut critical = Vec::new();
    let mut warnings = Vec::new();
    for s in steps {
        for a in &s.health.anomalies {
            match a.severity {
                ENHealthLevel::Critical => critical.push((s.step, a)),
                _ => warnings.push((s.step, a)),
            }
        }
    }

    if let Some((step, first)) = critical.first() {
        return (
            ENVerdict::Fail,
            format!(
                "{} critical anomal{} across {} step{}; first was {} at step {step}: {}",
                critical.len(),
                if critical.len() == 1 { "y" } else { "ies" },
                steps.len(),
                if steps.len() == 1 { "" } else { "s" },
                first.code,
                first.message
            ),
        );
    }
    if let Some((step, first)) = warnings.first() {
        return (
            ENVerdict::Warn,
            format!(
                "{} warning{} across {} step{}, none critical; first was {} at step {step}: {}",
                warnings.len(),
                if warnings.len() == 1 { "" } else { "s" },
                steps.len(),
                if steps.len() == 1 { "" } else { "s" },
                first.code,
                first.message
            ),
        );
    }
    (
        ENVerdict::Pass,
        format!(
            "no anomalies across {} step{}",
            steps.len(),
            if steps.len() == 1 { "" } else { "s" }
        ),
    )
}

/// One step as JSON.
fn step_json(s: &VLObservedStep) -> Json {
    let params: Vec<Json> = s
        .parameters
        .iter()
        .map(|p| {
            Json::Obj(BTreeMap::from([
                ("name".into(), Json::s(&p.name)),
                ("trainable".into(), Json::Bool(p.trainable)),
                ("grad_norm".into(), jnum_finite(p.grad_norm as f64)),
                ("grad_max".into(), jnum_finite(p.grad_max as f64)),
                ("grad_min".into(), jnum_finite(p.grad_min as f64)),
                ("grad_mean".into(), jnum_finite(p.grad_mean as f64)),
                ("update_norm".into(), jnum_finite(p.update_norm as f64)),
                ("update_max".into(), jnum_finite(p.update_max as f64)),
                ("weight_norm".into(), jnum_finite(p.weight_norm as f64)),
                (
                    "grad_norm_running_mean_5".into(),
                    jnum_finite(p.grad_norm_running_mean_5 as f64),
                ),
                (
                    "update_norm_running_mean_5".into(),
                    jnum_finite(p.update_norm_running_mean_5 as f64),
                ),
                (
                    "rows_updated".into(),
                    jnum_opt(p.rows_updated.map(|v| v as f64)),
                ),
                (
                    "rows_total".into(),
                    jnum_opt(p.rows_total.map(|v| v as f64)),
                ),
            ]))
        })
        .collect();

    let anomalies: Vec<Json> = s.health.anomalies.iter().map(anomaly_json).collect();

    Json::Obj(BTreeMap::from([
        ("step".into(), Json::n(s.step as f64)),
        ("epoch".into(), Json::n(s.epoch as f64)),
        ("lr".into(), Json::n(s.lr)),
        (
            "numerical".into(),
            Json::Obj(BTreeMap::from([
                ("has_nan_loss".into(), Json::Bool(s.numerical.has_nan_loss)),
                ("has_inf_loss".into(), Json::Bool(s.numerical.has_inf_loss)),
                ("has_nan_grad".into(), Json::Bool(s.numerical.has_nan_grad)),
                ("has_inf_grad".into(), Json::Bool(s.numerical.has_inf_grad)),
                (
                    "has_nan_param".into(),
                    Json::Bool(s.numerical.has_nan_param),
                ),
                (
                    "has_inf_param".into(),
                    Json::Bool(s.numerical.has_inf_param),
                ),
                (
                    "has_nan_optimizer_state".into(),
                    Json::Bool(s.numerical.has_nan_optimizer_state),
                ),
                (
                    "dead_params".into(),
                    Json::Arr(s.numerical.dead_params.iter().map(Json::s).collect()),
                ),
            ])),
        ),
        (
            "dynamics".into(),
            Json::Obj(BTreeMap::from([
                ("loss".into(), jnum_finite(s.dynamics.loss as f64)),
                (
                    "delta_loss".into(),
                    jnum_finite(s.dynamics.delta_loss as f64),
                ),
                (
                    "loss_ratio".into(),
                    jnum_finite(s.dynamics.loss_ratio as f64),
                ),
                (
                    "loss_running_mean_5".into(),
                    jnum_finite(s.dynamics.loss_running_mean_5 as f64),
                ),
                (
                    "perplexity".into(),
                    jnum_finite(s.dynamics.perplexity as f64),
                ),
                (
                    "supervised_token_count".into(),
                    Json::n(s.dynamics.supervised_token_count as f64),
                ),
                (
                    "total_token_count".into(),
                    Json::n(s.dynamics.total_token_count as f64),
                ),
                (
                    "supervised_ratio".into(),
                    jnum_finite(s.dynamics.supervised_ratio as f64),
                ),
                (
                    "batch_loss_variance".into(),
                    jnum_finite(s.dynamics.batch_loss_variance as f64),
                ),
            ])),
        ),
        ("per_param".into(), Json::Arr(params)),
        (
            "global_grad_norm".into(),
            jnum_finite(s.global_grad_norm as f64),
        ),
        (
            "global_update_norm".into(),
            jnum_finite(s.global_update_norm as f64),
        ),
        (
            "timing".into(),
            Json::Obj(BTreeMap::from([
                ("data_load_ms".into(), Json::n(s.timing.data_load_ms)),
                ("forward_ms".into(), Json::n(s.timing.forward_ms)),
                ("backward_ms".into(), Json::n(s.timing.backward_ms)),
                ("optimizer_ms".into(), Json::n(s.timing.optimizer_ms)),
                ("checkpoint_ms".into(), Json::n(s.timing.checkpoint_ms)),
                ("total_step_ms".into(), Json::n(s.timing.total_step_ms)),
                (
                    "supervised_tokens_per_sec".into(),
                    jnum_finite(s.timing.supervised_tokens_per_sec),
                ),
                (
                    "total_tokens_per_sec".into(),
                    jnum_finite(s.timing.total_tokens_per_sec),
                ),
                (
                    "forward_pct".into(),
                    jnum_finite(s.timing.forward_pct as f64),
                ),
                (
                    "backward_pct".into(),
                    jnum_finite(s.timing.backward_pct as f64),
                ),
                (
                    "optimizer_pct".into(),
                    jnum_finite(s.timing.optimizer_pct as f64),
                ),
                (
                    "step_time_median_ms".into(),
                    Json::n(s.timing.step_time_median_ms),
                ),
                (
                    "step_time_p95_ms".into(),
                    Json::n(s.timing.step_time_p95_ms),
                ),
            ])),
        ),
        (
            "memory".into(),
            Json::Obj(BTreeMap::from([
                (
                    "heap_rss_bytes".into(),
                    jnum_opt(s.memory.heap_rss_bytes.map(|v| v as f64)),
                ),
                (
                    "heap_delta_bytes".into(),
                    jnum_opt(s.memory.heap_delta_bytes.map(|v| v as f64)),
                ),
                (
                    "peak_rss_bytes".into(),
                    jnum_opt(s.memory.peak_rss_bytes.map(|v| v as f64)),
                ),
                ("heap_rss_mb".into(), jnum_opt(s.memory.heap_rss_mb())),
                (
                    "available_ram_bytes".into(),
                    jnum_opt(s.memory.available_ram_bytes.map(|v| v as f64)),
                ),
                (
                    "alloc_count_delta".into(),
                    jnum_opt(s.memory.alloc_count_delta.map(|v| v as f64)),
                ),
            ])),
        ),
        (
            "hardware".into(),
            Json::Obj(BTreeMap::from([
                (
                    "cpu_temp_celsius".into(),
                    jnum_opt(s.hardware.cpu_temp_celsius),
                ),
                ("cpu_freq_mhz".into(), jnum_opt(s.hardware.cpu_freq_mhz)),
                (
                    "cpu_freq_baseline_mhz".into(),
                    jnum_opt(s.hardware.cpu_freq_baseline_mhz),
                ),
                (
                    "throttle_detected".into(),
                    Json::Bool(s.hardware.throttle_detected),
                ),
                ("cpu_usage_pct".into(), jnum_opt(s.hardware.cpu_usage_pct)),
            ])),
        ),
        (
            "system".into(),
            Json::Obj(BTreeMap::from([
                (
                    "thread_count".into(),
                    jnum_opt(s.system.thread_count.map(|v| v as f64)),
                ),
                (
                    "voluntary_ctx_switches".into(),
                    jnum_opt(s.system.voluntary_ctx_switches.map(|v| v as f64)),
                ),
                (
                    "page_faults_major".into(),
                    jnum_opt(s.system.page_faults_major.map(|v| v as f64)),
                ),
                (
                    "page_faults_minor".into(),
                    jnum_opt(s.system.page_faults_minor.map(|v| v as f64)),
                ),
            ])),
        ),
        (
            "runtime".into(),
            Json::Obj(BTreeMap::from([
                (
                    "checkpoint_verified".into(),
                    s.runtime.checkpoint_verified.map_or(Json::Null, Json::Bool),
                ),
                (
                    "tape_nodes_after_backward".into(),
                    Json::n(s.runtime.tape_nodes_after_backward as f64),
                ),
                (
                    "optimizer_state_step".into(),
                    Json::n(s.runtime.optimizer_state_step as f64),
                ),
                (
                    "optimizer_stepped".into(),
                    Json::Bool(s.runtime.optimizer_stepped),
                ),
            ])),
        ),
        (
            "health".into(),
            Json::Obj(BTreeMap::from([
                ("level".into(), Json::s(s.health.level.as_str())),
                ("anomalies".into(), Json::Arr(anomalies)),
            ])),
        ),
    ]))
}

fn anomaly_json(a: &VLAnomaly) -> Json {
    Json::Obj(BTreeMap::from([
        ("code".into(), Json::s(a.code)),
        ("axis".into(), Json::s(a.axis)),
        ("param".into(), a.param.as_ref().map_or(Json::Null, Json::s)),
        ("message".into(), Json::s(&a.message)),
        ("severity".into(), Json::s(a.severity.as_str())),
    ]))
}

/// Steps that raised a given code, as a JSON array of step numbers.
fn steps_with(steps: &[VLObservedStep], code: &str) -> Json {
    Json::Arr(
        steps
            .iter()
            .filter(|s| s.health.anomalies.iter().any(|a| a.code == code))
            .map(|s| Json::n(s.step as f64))
            .collect(),
    )
}

/// The whole report as a `Json` document.
pub fn report_json(config: &VLRunConfig, steps: &[VLObservedStep]) -> Json {
    let (v, reason) = verdict(steps);

    let losses: Vec<f32> = steps.iter().map(|s| s.dynamics.loss).collect();
    let finite: Vec<f32> = losses.iter().copied().filter(|l| l.is_finite()).collect();
    let step_times: Vec<f64> = steps.iter().map(|s| s.timing.total_step_ms).collect();
    let mean_step = if step_times.is_empty() {
        0.0
    } else {
        step_times.iter().sum::<f64>() / step_times.len() as f64
    };

    // Per-parameter extremes, keyed by name so a report can be diffed across
    // runs without depending on parameter order.
    let mut per_param: BTreeMap<String, (f32, f32, f32)> = BTreeMap::new();
    for s in steps {
        for p in &s.parameters {
            let e = per_param.entry(p.name.clone()).or_insert((0.0, 0.0, 0.0));
            if p.grad_norm.is_finite() {
                e.0 = e.0.max(p.grad_norm);
            }
            if p.update_norm.is_finite() {
                e.1 = e.1.max(p.update_norm);
            }
            if p.weight_norm.is_finite() {
                e.2 = e.2.max(p.weight_norm);
            }
        }
    }
    let per_param_json = Json::Obj(
        per_param
            .into_iter()
            .map(|(name, (g, u, w))| {
                (
                    name,
                    Json::Obj(BTreeMap::from([
                        ("max_grad_norm".into(), jnum_finite(g as f64)),
                        ("max_update_norm".into(), jnum_finite(u as f64)),
                        ("max_weight_norm".into(), jnum_finite(w as f64)),
                    ])),
                )
            })
            .collect(),
    );

    let all_anomalies: Vec<Json> = steps
        .iter()
        .flat_map(|s| {
            s.health.anomalies.iter().map(move |a| {
                let mut m = match anomaly_json(a) {
                    Json::Obj(m) => m,
                    _ => BTreeMap::new(),
                };
                m.insert("step".into(), Json::n(s.step as f64));
                Json::Obj(m)
            })
        })
        .collect();

    let max_temp = steps
        .iter()
        .filter_map(|s| s.hardware.cpu_temp_celsius)
        .fold(None::<f32>, |acc, t| Some(acc.map_or(t, |a| a.max(t))));
    let peak_rss = steps.iter().filter_map(|s| s.memory.peak_rss_bytes).max();

    Json::Obj(BTreeMap::from([
        (
            "config".into(),
            Json::Obj(BTreeMap::from([
                ("model".into(), Json::s(&config.model)),
                ("dataset".into(), Json::s(&config.dataset)),
                ("batch_size".into(), Json::n(config.batch_size as f64)),
                ("lr".into(), Json::n(config.lr)),
                ("steps".into(), Json::n(config.steps as f64)),
                ("timestamp".into(), Json::s(&config.timestamp)),
            ])),
        ),
        (
            "steps".into(),
            Json::Arr(steps.iter().map(step_json).collect()),
        ),
        (
            "summary".into(),
            Json::Obj(BTreeMap::from([
                (
                    "loss_curve".into(),
                    Json::Arr(losses.iter().map(|l| jnum_finite(*l as f64)).collect()),
                ),
                (
                    "per_axis".into(),
                    Json::Obj(BTreeMap::from([
                        (
                            "numerical".into(),
                            Json::Obj(BTreeMap::from([
                                ("nan_steps".into(), steps_with(steps, "NAN_DETECTED")),
                                ("inf_steps".into(), steps_with(steps, "INF_DETECTED")),
                                (
                                    "explosion_steps".into(),
                                    steps_with(steps, "GRADIENT_EXPLOSION"),
                                ),
                            ])),
                        ),
                        (
                            "dynamics".into(),
                            Json::Obj(BTreeMap::from([
                                (
                                    "min_loss".into(),
                                    jnum_opt(finite.iter().copied().fold(None::<f32>, |a, l| {
                                        Some(a.map_or(l, |a: f32| a.min(l)))
                                    })),
                                ),
                                (
                                    "max_loss".into(),
                                    jnum_opt(finite.iter().copied().fold(None::<f32>, |a, l| {
                                        Some(a.map_or(l, |a: f32| a.max(l)))
                                    })),
                                ),
                                ("final_loss".into(), jnum_opt(losses.last().copied())),
                                ("plateau_steps".into(), steps_with(steps, "LOSS_PLATEAU")),
                                ("spike_steps".into(), steps_with(steps, "LOSS_SPIKE")),
                            ])),
                        ),
                        (
                            "parameters".into(),
                            Json::Obj(BTreeMap::from([("per_param".into(), per_param_json)])),
                        ),
                        (
                            "timing".into(),
                            Json::Obj(BTreeMap::from([
                                ("mean_step_ms".into(), Json::n(mean_step)),
                                (
                                    "p95_step_ms".into(),
                                    Json::n(
                                        steps.last().map_or(0.0, |s| s.timing.step_time_p95_ms),
                                    ),
                                ),
                                ("spike_steps".into(), steps_with(steps, "TIMING_SPIKE")),
                            ])),
                        ),
                        (
                            "memory".into(),
                            Json::Obj(BTreeMap::from([
                                (
                                    "peak_rss_mb".into(),
                                    jnum_opt(peak_rss.map(|b| b as f64 / 1048576.0)),
                                ),
                                ("leak_steps".into(), steps_with(steps, "MEMORY_LEAK")),
                            ])),
                        ),
                        (
                            "hardware".into(),
                            Json::Obj(BTreeMap::from([
                                (
                                    "throttle_steps".into(),
                                    steps_with(steps, "THERMAL_THROTTLE"),
                                ),
                                ("max_temp_celsius".into(), jnum_opt(max_temp)),
                            ])),
                        ),
                        (
                            "system".into(),
                            Json::Obj(BTreeMap::from([
                                (
                                    "ctx_switch_spike_steps".into(),
                                    steps_with(steps, "CTX_SWITCH_SPIKE"),
                                ),
                                (
                                    "major_fault_steps".into(),
                                    steps_with(steps, "MAJOR_PAGE_FAULT"),
                                ),
                            ])),
                        ),
                        (
                            "runtime".into(),
                            Json::Obj(BTreeMap::from([
                                ("tape_leak_steps".into(), steps_with(steps, "TAPE_LEAK")),
                                (
                                    "checkpoint_failures".into(),
                                    steps_with(steps, "CHECKPOINT_CORRUPTION"),
                                ),
                            ])),
                        ),
                    ])),
                ),
                ("all_anomalies".into(), Json::Arr(all_anomalies)),
                ("verdict".into(), Json::s(v.as_str())),
                ("verdict_reason".into(), Json::s(reason)),
            ])),
        ),
    ]))
}

/// The Markdown report.
pub fn report_markdown(config: &VLRunConfig, steps: &[VLObservedStep]) -> String {
    let (v, reason) = verdict(steps);
    let mut out = String::new();

    out.push_str("# gltrain observability report\n\n");

    out.push_str("## 1. Config\n\n");
    out.push_str("| key | value |\n|---|---|\n");
    out.push_str(&format!("| model | {} |\n", config.model));
    out.push_str(&format!("| dataset | {} |\n", config.dataset));
    out.push_str(&format!("| batch_size | {} |\n", config.batch_size));
    out.push_str(&format!("| lr | {} |\n", config.lr));
    out.push_str(&format!("| steps | {} |\n", config.steps));
    out.push_str(&format!("| timestamp | {} |\n\n", config.timestamp));

    out.push_str("## 2. Loss curve\n\n");
    out.push_str("| step | loss | delta | perplexity | health |\n|---|---|---|---|---|\n");
    for s in steps {
        let ppl = if s.dynamics.perplexity.is_finite() {
            format!("{:.3}", s.dynamics.perplexity)
        } else {
            "inf".to_string()
        };
        out.push_str(&format!(
            "| {} | {:.6} | {:+.6} | {} | {} |\n",
            s.step,
            s.dynamics.loss,
            s.dynamics.delta_loss,
            ppl,
            s.health.level.as_str()
        ));
    }
    out.push('\n');

    out.push_str("## 3. Per-axis summary\n\n");
    let step_times: Vec<f64> = steps.iter().map(|s| s.timing.total_step_ms).collect();
    let mean_step = if step_times.is_empty() {
        0.0
    } else {
        step_times.iter().sum::<f64>() / step_times.len() as f64
    };
    out.push_str("| axis | summary |\n|---|---|\n");
    out.push_str(&format!(
        "| NUMERICAL | NaN steps {}, Inf steps {} |\n",
        count_code(steps, "NAN_DETECTED"),
        count_code(steps, "INF_DETECTED")
    ));
    let finals: Vec<f32> = steps.iter().map(|s| s.dynamics.loss).collect();
    out.push_str(&format!(
        "| DYNAMICS | first {:.6}, final {:.6}, plateau steps {} |\n",
        finals.first().copied().unwrap_or(0.0),
        finals.last().copied().unwrap_or(0.0),
        count_code(steps, "LOSS_PLATEAU")
    ));
    out.push_str(&format!(
        "| PARAMETER_HEALTH | {} parameters tracked, {} grad spikes |\n",
        steps.first().map_or(0, |s| s.parameters.len()),
        count_code(steps, "GRAD_SPIKE")
    ));
    out.push_str(&format!(
        "| TIMING | mean {mean_step:.1}ms, p95 {:.1}ms, {} spikes |\n",
        steps.last().map_or(0.0, |s| s.timing.step_time_p95_ms),
        count_code(steps, "TIMING_SPIKE")
    ));
    out.push_str(&format!(
        "| MEMORY | peak {}, {} leak steps |\n",
        steps
            .iter()
            .filter_map(|s| s.memory.peak_rss_bytes)
            .max()
            .map_or("n/a".to_string(), |b| format!(
                "{:.1}MB",
                b as f64 / 1048576.0
            )),
        count_code(steps, "MEMORY_LEAK")
    ));
    let max_temp = steps
        .iter()
        .filter_map(|s| s.hardware.cpu_temp_celsius)
        .fold(None::<f32>, |a, t| Some(a.map_or(t, |a| a.max(t))));
    out.push_str(&format!(
        "| HARDWARE | max temp {}, {} throttle steps |\n",
        opt_f32(max_temp, "C", 1),
        count_code(steps, "THERMAL_THROTTLE")
    ));
    out.push_str(&format!(
        "| SYSTEM | {} ctx-switch spikes, {} major-fault steps |\n",
        count_code(steps, "CTX_SWITCH_SPIKE"),
        count_code(steps, "MAJOR_PAGE_FAULT")
    ));
    out.push_str(&format!(
        "| RUNTIME | {} tape leaks, {} checkpoint failures |\n\n",
        count_code(steps, "TAPE_LEAK"),
        count_code(steps, "CHECKPOINT_CORRUPTION")
    ));

    out.push_str("## 4. Anomaly log\n\n");
    let total: usize = steps.iter().map(|s| s.health.anomalies.len()).sum();
    if total == 0 {
        out.push_str("None.\n\n");
    } else {
        out.push_str("| step | axis | code | severity | message |\n|---|---|---|---|---|\n");
        for s in steps {
            for a in &s.health.anomalies {
                out.push_str(&format!(
                    "| {} | {} | {} | {} | {} |\n",
                    s.step,
                    a.axis,
                    a.code,
                    a.severity.as_str(),
                    a.message
                ));
            }
        }
        out.push('\n');
    }

    out.push_str("## 5. Verdict\n\n");
    out.push_str(&format!("**{}** — {}\n", v.as_str(), reason));
    out
}

fn count_code(steps: &[VLObservedStep], code: &str) -> usize {
    steps
        .iter()
        .flat_map(|s| s.health.anomalies.iter())
        .filter(|a| a.code == code)
        .count()
}

/// Write both report files into `dir`, returning their paths as
/// `(json, markdown)`.
pub fn write_reports(
    dir: &Path,
    config: &VLRunConfig,
    steps: &[VLObservedStep],
) -> Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(dir)?;
    let stem = format!("gltrain_obs_{}", config.timestamp);
    let json_path = dir.join(format!("{stem}.json"));
    let md_path = dir.join(format!("{stem}.md"));

    let doc = report_json(config, steps).to_compact();
    // Parsing back what was just written is cheap and catches an encoder bug
    // at the moment it happens, rather than when someone tries to read the
    // report days later.
    json::parse(&doc).map_err(|e| {
        GlTrainError::Data(format!(
            "the report this run just wrote is not valid JSON: {e}"
        ))
    })?;

    std::fs::write(&json_path, doc)?;
    std::fs::write(&md_path, report_markdown(config, steps))?;
    Ok((json_path, md_path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::train::observability::anomaly::{VLAnomaly, VLStepHealth};
    use crate::train::observability::axes::{VLDynamics, VLParamHealth, VLTiming};

    fn step(n: usize, loss: f32, anomalies: Vec<VLAnomaly>) -> VLObservedStep {
        VLObservedStep {
            step: n,
            epoch: 0,
            lr: 1e-3,
            numerical: Default::default(),
            dynamics: VLDynamics {
                loss,
                perplexity: loss.exp(),
                total_token_count: 100,
                supervised_token_count: 80,
                supervised_ratio: 0.8,
                ..Default::default()
            },
            parameters: vec![VLParamHealth {
                name: "embed.weight".into(),
                trainable: true,
                grad_norm: 0.5,
                update_norm: 0.1,
                weight_norm: 2.0,
                ..Default::default()
            }],
            global_grad_norm: 0.5,
            global_update_norm: 0.1,
            timing: VLTiming {
                total_step_ms: 10.0,
                forward_ms: 5.0,
                backward_ms: 4.0,
                optimizer_ms: 1.0,
                ..Default::default()
            },
            memory: Default::default(),
            hardware: Default::default(),
            system: Default::default(),
            runtime: Default::default(),
            health: VLStepHealth::from_anomalies(anomalies),
        }
    }

    fn config() -> VLRunConfig {
        VLRunConfig {
            model: "ABEmbedding + ABLinear".into(),
            dataset: "gwen_code_dataset.jsonl".into(),
            batch_size: 4,
            lr: 0.02,
            steps: 3,
            timestamp: "1700000000".into(),
        }
    }

    // ── Verdict ──────────────────────────────────────────────────────────

    #[test]
    fn a_clean_run_passes() {
        let steps = vec![step(1, 5.0, vec![]), step(2, 4.0, vec![])];
        let (v, reason) = verdict(&steps);
        assert_eq!(v, ENVerdict::Pass);
        assert!(reason.contains("no anomalies"), "{reason}");
    }

    #[test]
    fn a_warning_downgrades_the_run_to_warn_but_not_fail() {
        let steps = vec![
            step(1, 5.0, vec![]),
            step(
                2,
                4.0,
                vec![VLAnomaly::warning("GRAD_SPIKE", "PARAMETER_HEALTH", "m")],
            ),
        ];
        let (v, reason) = verdict(&steps);
        assert_eq!(v, ENVerdict::Warn);
        assert!(reason.contains("GRAD_SPIKE"), "{reason}");
        assert!(reason.contains("step 2"), "{reason}");
    }

    /// One critical anomaly fails the run regardless of how many warnings
    /// surround it, and the reason names the first one.
    #[test]
    fn one_critical_anomaly_fails_the_whole_run() {
        let steps = vec![
            step(
                1,
                5.0,
                vec![VLAnomaly::warning("GRAD_SPIKE", "PARAMETER_HEALTH", "m")],
            ),
            step(
                2,
                f32::NAN,
                vec![VLAnomaly::critical(
                    "NAN_DETECTED",
                    "NUMERICAL",
                    "NaN in loss",
                )],
            ),
        ];
        let (v, reason) = verdict(&steps);
        assert_eq!(v, ENVerdict::Fail);
        assert!(reason.contains("NAN_DETECTED"), "{reason}");
    }

    // ── JSON ─────────────────────────────────────────────────────────────

    /// The report must parse. It is written by this crate's encoder and read
    /// by this crate's parser, so a document that does not round-trip is a bug
    /// in a path the checkpoint format shares.
    #[test]
    fn the_json_report_is_valid_and_carries_every_axis() {
        let steps = vec![step(1, 5.0, vec![]), step(2, 4.0, vec![])];
        let doc = report_json(&config(), &steps).to_compact();
        let parsed = json::parse(&doc).expect("the report must be valid JSON");

        let root = parsed.as_obj().expect("object");
        assert!(root.contains_key("config"));
        assert!(root.contains_key("steps"));
        assert!(root.contains_key("summary"));

        let first = &parsed.get("steps").unwrap().as_arr().unwrap()[0];
        for axis in [
            "numerical",
            "dynamics",
            "per_param",
            "timing",
            "memory",
            "hardware",
            "system",
            "runtime",
            "health",
        ] {
            assert!(first.get(axis).is_some(), "step JSON is missing {axis}");
        }
        assert_eq!(
            parsed
                .get("summary")
                .unwrap()
                .get("verdict")
                .unwrap()
                .as_str(),
            Some("PASS")
        );
    }

    /// JSON has no NaN literal. A NaN loss is exactly what this report exists
    /// to record, so it must become `null` and leave the document parseable
    /// rather than producing a file nothing can read.
    #[test]
    fn a_nan_loss_serializes_as_null_and_keeps_the_document_parseable() {
        let steps = vec![step(
            1,
            f32::NAN,
            vec![VLAnomaly::critical(
                "NAN_DETECTED",
                "NUMERICAL",
                "NaN in loss",
            )],
        )];
        let doc = report_json(&config(), &steps).to_compact();
        // Only a *bare* NaN is a problem. The word appears legitimately inside
        // the anomaly message, which is a quoted string and parses fine.
        for bare in [":NaN", ",NaN", "[NaN", ":inf", ",inf", "[inf", ":-inf"] {
            assert!(
                !doc.contains(bare),
                "a bare {bare} literal leaked into the JSON"
            );
        }
        let parsed = json::parse(&doc).expect("must still parse");
        assert_eq!(
            parsed
                .get("summary")
                .unwrap()
                .get("verdict")
                .unwrap()
                .as_str(),
            Some("FAIL")
        );
    }

    // ── Markdown ─────────────────────────────────────────────────────────

    #[test]
    fn the_markdown_report_carries_the_verdict_and_the_anomaly_table() {
        let steps = vec![
            step(1, 5.0, vec![]),
            step(
                2,
                4.0,
                vec![VLAnomaly::warning(
                    "GRAD_SPIKE",
                    "PARAMETER_HEALTH",
                    "embed.weight spiked",
                )],
            ),
        ];
        let md = report_markdown(&config(), &steps);
        for section in [
            "## 1. Config",
            "## 2. Loss curve",
            "## 3. Per-axis summary",
            "## 4. Anomaly log",
            "## 5. Verdict",
        ] {
            assert!(md.contains(section), "missing {section}");
        }
        assert!(md.contains("**WARN**"));
        assert!(md.contains("GRAD_SPIKE"));
        assert!(md.contains("embed.weight spiked"));
    }

    #[test]
    fn a_clean_markdown_report_says_the_anomaly_log_is_empty() {
        let md = report_markdown(&config(), &[step(1, 5.0, vec![])]);
        assert!(md.contains("None."));
        assert!(md.contains("**PASS**"));
    }

    // ── Live line ────────────────────────────────────────────────────────

    /// Absent readings print as `n/a`, never as a zero that reads as healthy.
    #[test]
    fn the_live_line_prints_na_for_unmeasured_hardware() {
        let line = live_line(&step(3, 5.438, vec![]), 10);
        assert!(line.contains("Step   3/10"), "{line}");
        assert!(line.contains("loss 5.4380"), "{line}");
        assert!(
            line.contains("n/a"),
            "unmeasured fields must read n/a: {line}"
        );
        assert!(line.ends_with("OK"), "{line}");
    }

    #[test]
    fn anomaly_lines_name_their_axis_and_code() {
        let s = step(
            5,
            4.9,
            vec![VLAnomaly::warning(
                "THERMAL_THROTTLE",
                "HARDWARE",
                "CPU at 2600MHz",
            )],
        );
        let lines = anomaly_lines(&s);
        assert_eq!(lines.len(), 1);
        assert!(
            lines[0].contains("[HARDWARE/THERMAL_THROTTLE]"),
            "{}",
            lines[0]
        );
    }

    // ── Files ────────────────────────────────────────────────────────────

    #[test]
    fn write_reports_produces_both_files_and_the_json_parses_from_disk() {
        let dir = std::env::temp_dir().join("gltrain_obs_test_write");
        let steps = vec![step(1, 5.0, vec![])];
        let (json_path, md_path) = write_reports(&dir, &config(), &steps).expect("write");

        let doc = std::fs::read_to_string(&json_path).expect("json readable");
        json::parse(&doc).expect("json on disk must parse");
        let md = std::fs::read_to_string(&md_path).expect("md readable");
        assert!(md.contains("**PASS**"));

        let _ = std::fs::remove_file(&json_path);
        let _ = std::fs::remove_file(&md_path);
    }
}
