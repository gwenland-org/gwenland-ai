//! Layer scheduling and adaptive prefetch (ARTX05 §"Runtime Scheduler").
//!
//! Execution is strictly sequential — layer `N` completes before layer `N+1`
//! starts. The scheduler's only freedom is *when to map* a layer: while layer
//! `N` executes, layers `N+1 ..= N+W` may already be mapped.
//!
//! `W` is not fixed. It grows when mapping is the bottleneck and shrinks when
//! it is not, so a fast NVMe and a slow spinning disk converge on different
//! windows without the caller tuning anything.

use std::collections::VecDeque;
use std::time::Duration;

use crate::runtime::types::RuntimeConfig;

/// How many recent layers feed the moving averages.
///
/// Eight is roughly a quarter of a 28-layer model: long enough to absorb a
/// single slow read, short enough to react within one pass.
const TIMING_HISTORY: usize = 8;

/// Grow the window when loading takes more than this multiple of exec time.
const GROW_RATIO: f64 = 1.2;

/// Shrink the window when loading takes less than this multiple of exec time.
const SHRINK_RATIO: f64 = 0.5;

/// Per-layer progress through one forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayerScheduleState {
    /// Not yet mapped.
    #[default]
    Pending,
    /// Mapped and ready to execute.
    Ready,
    /// Executed successfully.
    Done,
    /// Execution failed.
    Failed,
}

/// Moving-average controller for the prefetch window (ARTX06 §5.1).
///
/// Compares how long layers take to *map* against how long they take to
/// *execute*. Mapping slower than execution means the runtime is I/O-starved
/// and should read further ahead; the reverse means memory is being pinned
/// for no gain.
#[derive(Debug, Clone)]
pub struct AdaptivePrefetcher {
    window_size: usize,
    max_window: usize,
    load_times: VecDeque<Duration>,
    exec_times: VecDeque<Duration>,
}

impl AdaptivePrefetcher {
    /// Start at `initial` window, never exceeding `max_window`.
    ///
    /// A `max_window` of 0 pins the window at 0, disabling prefetch entirely.
    pub fn new(initial: usize, max_window: usize) -> Self {
        AdaptivePrefetcher {
            window_size: initial.min(max_window),
            max_window,
            load_times: VecDeque::with_capacity(TIMING_HISTORY),
            exec_times: VecDeque::with_capacity(TIMING_HISTORY),
        }
    }

    /// Derive `max_window` from a memory budget and typical layer size, then
    /// build a prefetcher (ARTX05 §"Prefetch Window").
    ///
    /// Returns a window of 0 when a single layer does not fit in the budget —
    /// prefetching would then be a guaranteed eviction.
    pub fn with_memory_budget(initial: usize, available_bytes: u64, layer_bytes: u64) -> Self {
        let max = match available_bytes.checked_div(layer_bytes) {
            Some(fits) => usize::try_from(fits).unwrap_or(usize::MAX),
            // Unknown layer size: keep the caller's window rather than
            // inventing a ceiling from a division by zero.
            None => initial,
        };
        Self::new(initial, max)
    }

    /// Record how long a layer took to map.
    pub fn record_load_time(&mut self, d: Duration) {
        push_capped(&mut self.load_times, d);
    }

    /// Record how long a layer took to execute.
    pub fn record_exec_time(&mut self, d: Duration) {
        push_capped(&mut self.exec_times, d);
    }

    /// Re-evaluate the window against the current moving averages.
    ///
    /// No-op until both averages have a sample: without an exec time there is
    /// no baseline to compare against, and guessing would oscillate.
    pub fn adjust_window(&mut self) {
        let (Some(load), Some(exec)) = (mean(&self.load_times), mean(&self.exec_times)) else {
            return;
        };
        // A zero exec average means the backend is a stub (or the timer is
        // coarser than the work). Treat load as dominant and read further
        // ahead rather than dividing by zero.
        if exec == 0.0 {
            if load > 0.0 {
                self.grow();
            }
            return;
        }
        let ratio = load / exec;
        if ratio > GROW_RATIO {
            self.grow();
        } else if ratio < SHRINK_RATIO {
            self.shrink();
        }
    }

    fn grow(&mut self) {
        self.window_size = (self.window_size + 1).min(self.max_window);
    }

    fn shrink(&mut self) {
        // Floor of 1 keeps one layer in flight; dropping to 0 would serialize
        // mapping behind execution. When max_window is 0 prefetch is off by
        // configuration and must stay off.
        let floor = self.max_window.min(1);
        self.window_size = self.window_size.saturating_sub(1).max(floor);
    }

    /// Current window size.
    pub fn current_window(&self) -> usize {
        self.window_size
    }

    /// Upper bound the window may reach.
    pub fn max_window(&self) -> usize {
        self.max_window
    }

    /// Mean load time over the history, `None` until a sample exists.
    pub fn avg_load_time(&self) -> Option<Duration> {
        mean(&self.load_times).map(Duration::from_secs_f64)
    }

    /// Mean exec time over the history, `None` until a sample exists.
    pub fn avg_exec_time(&self) -> Option<Duration> {
        mean(&self.exec_times).map(Duration::from_secs_f64)
    }
}

fn push_capped(q: &mut VecDeque<Duration>, d: Duration) {
    if q.len() == TIMING_HISTORY {
        q.pop_front();
    }
    q.push_back(d);
}

fn mean(q: &VecDeque<Duration>) -> Option<f64> {
    if q.is_empty() {
        return None;
    }
    let total: f64 = q.iter().map(|d| d.as_secs_f64()).sum();
    Some(total / q.len() as f64)
}

/// Drives layers through `Pending → Ready → Done` in order, and decides which
/// layers to map ahead.
#[derive(Debug)]
pub struct Scheduler {
    total_layers: u32,
    current_layer: u32,
    states: Vec<LayerScheduleState>,
    prefetcher: AdaptivePrefetcher,
}

impl Scheduler {
    /// Build a scheduler for `total_layers`, seeded from `config`.
    pub fn new(total_layers: u32, config: &RuntimeConfig) -> Self {
        Self::with_prefetcher(
            total_layers,
            AdaptivePrefetcher::new(config.prefetch_window, config.prefetch_window.max(1)),
        )
    }

    /// Build a scheduler with an explicitly configured prefetcher — used when
    /// the window's ceiling comes from a memory budget rather than config.
    pub fn with_prefetcher(total_layers: u32, prefetcher: AdaptivePrefetcher) -> Self {
        Scheduler {
            total_layers,
            current_layer: 0,
            states: vec![LayerScheduleState::Pending; total_layers as usize],
            prefetcher,
        }
    }

    /// Index of the layer awaiting execution, or `None` when the pass is done.
    pub fn current(&self) -> Option<u32> {
        (self.current_layer < self.total_layers).then_some(self.current_layer)
    }

    /// Mark the current layer `Done` and move to the next.
    ///
    /// Returns the new current layer, or `None` if the pass just completed.
    pub fn advance(&mut self) -> Option<u32> {
        if self.current_layer < self.total_layers {
            self.states[self.current_layer as usize] = LayerScheduleState::Done;
            self.current_layer += 1;
        }
        self.current()
    }

    /// Layers that should be mapped now: the next `W` still-pending layers
    /// after the current one, clamped to the end of the model.
    pub fn layers_to_prefetch(&self) -> Vec<u32> {
        let window = self.prefetcher.current_window();
        if window == 0 {
            return Vec::new();
        }
        let start = self.current_layer.saturating_add(1);
        (start..self.total_layers)
            .take(window)
            .filter(|&i| self.states[i as usize] == LayerScheduleState::Pending)
            .collect()
    }

    /// Note that a layer is mapped and ready.
    pub fn mark_ready(&mut self, index: u32) {
        self.set_state(index, LayerScheduleState::Ready);
    }

    /// Note that a layer executed successfully.
    pub fn mark_done(&mut self, index: u32) {
        self.set_state(index, LayerScheduleState::Done);
    }

    /// Note that a layer failed.
    pub fn mark_failed(&mut self, index: u32) {
        self.set_state(index, LayerScheduleState::Failed);
    }

    fn set_state(&mut self, index: u32, state: LayerScheduleState) {
        if let Some(slot) = self.states.get_mut(index as usize) {
            *slot = state;
        }
    }

    /// State of one layer, `None` if out of range.
    pub fn state_of(&self, index: u32) -> Option<LayerScheduleState> {
        self.states.get(index as usize).copied()
    }

    /// Whether a layer is mapped and ready to execute.
    pub fn is_ready(&self, index: u32) -> bool {
        self.state_of(index) == Some(LayerScheduleState::Ready)
    }

    /// Whether every layer reached `Done`.
    ///
    /// A model with zero layers is trivially complete.
    pub fn is_complete(&self) -> bool {
        self.states.iter().all(|s| *s == LayerScheduleState::Done)
    }

    /// Whether any layer failed.
    pub fn has_failure(&self) -> bool {
        self.states.contains(&LayerScheduleState::Failed)
    }

    /// Feed one layer's timings to the prefetcher and re-evaluate the window.
    pub fn record_timings(&mut self, load: Duration, exec: Duration) {
        self.prefetcher.record_load_time(load);
        self.prefetcher.record_exec_time(exec);
        self.prefetcher.adjust_window();
    }

    /// Current prefetch window.
    pub fn prefetch_window(&self) -> usize {
        self.prefetcher.current_window()
    }

    /// Read-only access to the prefetcher (timings, bounds).
    pub fn prefetcher(&self) -> &AdaptivePrefetcher {
        &self.prefetcher
    }

    /// Total layers in this pass.
    pub fn total_layers(&self) -> u32 {
        self.total_layers
    }

    /// Return every layer to `Pending` and rewind to layer 0.
    pub fn reset(&mut self) {
        self.current_layer = 0;
        self.states.fill(LayerScheduleState::Pending);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config_with_window(w: usize) -> RuntimeConfig {
        RuntimeConfig::default().with_prefetch_window(w)
    }

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    // --- Scheduler ---------------------------------------------------------

    #[test]
    fn advances_through_layers_in_order() {
        let mut s = Scheduler::new(3, &config_with_window(2));
        assert_eq!(s.current(), Some(0));
        assert_eq!(s.advance(), Some(1));
        assert_eq!(s.advance(), Some(2));
        assert_eq!(s.advance(), None, "past the last layer");
        assert!(s.is_complete());
    }

    #[test]
    fn advancing_past_the_end_is_saturating() {
        let mut s = Scheduler::new(1, &config_with_window(1));
        assert_eq!(s.advance(), None);
        assert_eq!(s.advance(), None, "must not panic or wrap");
        assert!(s.is_complete());
    }

    #[test]
    fn zero_layer_model_is_immediately_complete() {
        let s = Scheduler::new(0, &config_with_window(2));
        assert!(s.is_complete());
        assert_eq!(s.current(), None);
        assert!(s.layers_to_prefetch().is_empty());
    }

    #[test]
    fn prefetch_list_is_clamped_at_the_last_layer() {
        let mut s = Scheduler::new(4, &config_with_window(2));
        assert_eq!(s.layers_to_prefetch(), vec![1, 2]);
        s.advance(); // now at 1
        s.advance(); // now at 2
        assert_eq!(s.layers_to_prefetch(), vec![3], "only layer 3 remains");
        s.advance(); // now at 3, the last
        assert!(
            s.layers_to_prefetch().is_empty(),
            "nothing to prefetch beyond the final layer"
        );
    }

    #[test]
    fn window_of_zero_disables_prefetch() {
        let s = Scheduler::with_prefetcher(4, AdaptivePrefetcher::new(0, 0));
        assert_eq!(s.prefetch_window(), 0);
        assert!(s.layers_to_prefetch().is_empty());
    }

    #[test]
    fn already_ready_layers_are_not_prefetched_twice() {
        let mut s = Scheduler::new(4, &config_with_window(3));
        assert_eq!(s.layers_to_prefetch(), vec![1, 2, 3]);
        s.mark_ready(2);
        assert_eq!(
            s.layers_to_prefetch(),
            vec![1, 3],
            "layer 2 is already mapped"
        );
    }

    #[test]
    fn tracks_ready_done_and_failed_states() {
        let mut s = Scheduler::new(3, &config_with_window(1));
        assert_eq!(s.state_of(0), Some(LayerScheduleState::Pending));

        s.mark_ready(0);
        assert!(s.is_ready(0));
        s.mark_done(0);
        assert_eq!(s.state_of(0), Some(LayerScheduleState::Done));
        assert!(!s.is_ready(0));

        s.mark_failed(1);
        assert!(s.has_failure());
        assert!(!s.is_complete(), "a failed layer is not complete");
        assert_eq!(s.state_of(99), None, "out of range");
    }

    #[test]
    fn marking_out_of_range_layers_is_ignored() {
        let mut s = Scheduler::new(2, &config_with_window(1));
        s.mark_ready(99); // must not panic
        s.mark_done(99);
        assert_eq!(s.state_of(99), None);
    }

    #[test]
    fn reset_rewinds_every_layer() {
        let mut s = Scheduler::new(3, &config_with_window(1));
        s.advance();
        s.mark_failed(2);
        s.reset();
        assert_eq!(s.current(), Some(0));
        assert!(!s.has_failure());
        assert_eq!(s.state_of(2), Some(LayerScheduleState::Pending));
    }

    // --- AdaptivePrefetcher ------------------------------------------------

    #[test]
    fn window_grows_when_loading_dominates() {
        let mut p = AdaptivePrefetcher::new(2, 8);
        p.record_load_time(ms(100));
        p.record_exec_time(ms(10)); // ratio 10.0 > 1.2
        p.adjust_window();
        assert_eq!(p.current_window(), 3);
    }

    #[test]
    fn window_shrinks_when_execution_dominates() {
        let mut p = AdaptivePrefetcher::new(4, 8);
        p.record_load_time(ms(10));
        p.record_exec_time(ms(100)); // ratio 0.1 < 0.5
        p.adjust_window();
        assert_eq!(p.current_window(), 3);
    }

    #[test]
    fn window_holds_inside_the_hysteresis_band() {
        let mut p = AdaptivePrefetcher::new(3, 8);
        p.record_load_time(ms(80));
        p.record_exec_time(ms(100)); // ratio 0.8: between 0.5 and 1.2
        p.adjust_window();
        assert_eq!(p.current_window(), 3, "no change inside the band");
    }

    #[test]
    fn window_is_clamped_to_max() {
        let mut p = AdaptivePrefetcher::new(2, 3);
        for _ in 0..10 {
            p.record_load_time(ms(100));
            p.record_exec_time(ms(1));
            p.adjust_window();
        }
        assert_eq!(p.current_window(), 3, "must not exceed max_window");
    }

    #[test]
    fn window_never_shrinks_below_one() {
        let mut p = AdaptivePrefetcher::new(3, 8);
        for _ in 0..10 {
            p.record_load_time(ms(1));
            p.record_exec_time(ms(1000));
            p.adjust_window();
        }
        assert_eq!(p.current_window(), 1, "one layer stays in flight");
    }

    #[test]
    fn disabled_prefetch_stays_disabled() {
        // max_window 0 means "never prefetch" — shrink must not raise it to 1.
        let mut p = AdaptivePrefetcher::new(0, 0);
        p.record_load_time(ms(1));
        p.record_exec_time(ms(1000));
        p.adjust_window();
        assert_eq!(p.current_window(), 0);
    }

    #[test]
    fn no_adjustment_before_both_samples_exist() {
        let mut p = AdaptivePrefetcher::new(2, 8);
        p.adjust_window();
        assert_eq!(p.current_window(), 2, "no samples at all");

        p.record_load_time(ms(100));
        p.adjust_window();
        assert_eq!(p.current_window(), 2, "load time alone is not a baseline");
    }

    #[test]
    fn zero_exec_time_grows_instead_of_dividing_by_zero() {
        let mut p = AdaptivePrefetcher::new(1, 4);
        p.record_load_time(ms(50));
        p.record_exec_time(Duration::ZERO);
        p.adjust_window();
        assert_eq!(p.current_window(), 2, "load dominates a null backend");
    }

    #[test]
    fn history_is_capped_and_drops_oldest_samples() {
        let mut p = AdaptivePrefetcher::new(2, 8);
        // Fill the history with slow loads, then flood it with fast ones.
        for _ in 0..TIMING_HISTORY {
            p.record_load_time(ms(100));
        }
        for _ in 0..TIMING_HISTORY {
            p.record_load_time(ms(10));
        }
        let avg = p.avg_load_time().unwrap();
        assert_eq!(
            avg,
            ms(10),
            "the 100ms samples must have aged out of the window"
        );
    }

    #[test]
    fn averages_are_none_until_sampled() {
        let p = AdaptivePrefetcher::new(2, 8);
        assert!(p.avg_load_time().is_none());
        assert!(p.avg_exec_time().is_none());
    }

    #[test]
    fn memory_budget_sets_the_ceiling() {
        // 1 GB budget, 100 MB layers => at most 10 layers in flight.
        let p = AdaptivePrefetcher::with_memory_budget(2, 1_000_000_000, 100_000_000);
        assert_eq!(p.max_window(), 10);
        assert_eq!(p.current_window(), 2);
    }

    #[test]
    fn a_layer_too_big_for_the_budget_disables_prefetch() {
        // Budget smaller than one layer: prefetching guarantees eviction.
        let p = AdaptivePrefetcher::with_memory_budget(2, 50_000_000, 100_000_000);
        assert_eq!(p.max_window(), 0);
        assert_eq!(p.current_window(), 0);
    }

    #[test]
    fn unknown_layer_size_keeps_the_requested_window() {
        // layer_bytes == 0 would divide by zero; the window must survive
        // intact rather than collapsing to 0 or saturating to usize::MAX.
        let p = AdaptivePrefetcher::with_memory_budget(2, 1_000_000_000, 0);
        assert_eq!(p.max_window(), 2);
        assert_eq!(p.current_window(), 2);
    }

    #[test]
    fn initial_window_is_capped_by_max_on_construction() {
        let p = AdaptivePrefetcher::new(10, 3);
        assert_eq!(p.current_window(), 3, "initial must respect max");
    }
}
