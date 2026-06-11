use gwenland_core::hardware::HardwareProfile;
use gwenland_core::tokenizer::{auto_estimate_context, get_active_model_from_config, TokenEstimate};
use gwenland_core::windowing::WindowConfig;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::panes::chat_pane::ChatPane;

const HISTORY_LEN: usize = 60;

pub struct App {
    #[allow(dead_code)]
    pub hardware: HardwareProfile,
    pub should_quit: bool,
    pub token_estimate: Arc<Mutex<Option<TokenEstimate>>>,
    pub active_model: Option<String>,
    pub live_monitor: gwenland_core::hardware::LiveMonitor,
    pub show_hw_panel: bool,
    pub ram_history: std::collections::VecDeque<f64>,
    pub cpu_history: std::collections::VecDeque<f64>,
    pub gpu_history: Vec<std::collections::VecDeque<f64>>,
    pub vram_history: Vec<std::collections::VecDeque<f64>>,
    pub latest_sample: Option<gwenland_core::hardware::UsageSample>,
    /// Owns all chat state: messages, input, streaming, rx channel.
    pub chat_pane: ChatPane,
}

impl App {
    pub fn new(hardware: HardwareProfile) -> Self {
        let token_estimate = Arc::new(Mutex::new(None));
        let estimate_clone = Arc::clone(&token_estimate);

        thread::spawn(move || {
            let current_dir = std::env::current_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("."));
            let ignore = gwenland_core::ignore_rules::load_ignore_rules(&current_dir);
            let scan_res = gwenland_core::scanner::scan_workspace(&current_dir, Some(&ignore));
            let estimate = auto_estimate_context(&scan_res.files, "");
            let mut lock = estimate_clone.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            *lock = Some(estimate);
        });

        let active_model = get_active_model_from_config();

        let mut ram_history = std::collections::VecDeque::with_capacity(HISTORY_LEN);
        let mut cpu_history = std::collections::VecDeque::with_capacity(HISTORY_LEN);
        for _ in 0..HISTORY_LEN {
            ram_history.push_back(0.0);
            cpu_history.push_back(0.0);
        }

        let mut gpu_history = Vec::new();
        let mut vram_history = Vec::new();
        for _ in 0..hardware.gpus.len() {
            let mut gpu_h = std::collections::VecDeque::with_capacity(HISTORY_LEN);
            let mut vram_h = std::collections::VecDeque::with_capacity(HISTORY_LEN);
            for _ in 0..HISTORY_LEN {
                gpu_h.push_back(0.0);
                vram_h.push_back(0.0);
            }
            gpu_history.push(gpu_h);
            vram_history.push(vram_h);
        }

        Self {
            hardware,
            should_quit: false,
            token_estimate,
            active_model,
            live_monitor: gwenland_core::hardware::LiveMonitor::new(),
            show_hw_panel: true,
            ram_history,
            cpu_history,
            gpu_history,
            vram_history,
            latest_sample: None,
            chat_pane: ChatPane::new(WindowConfig::load()),
        }
    }

    /// Append a character to the chat input (max 4096 chars).
    pub fn push_char(&mut self, c: char) {
        self.chat_pane.push_char(c);
    }

    pub fn pop_char(&mut self) {
        self.chat_pane.pop_char();
    }

    /// Submit the current chat input and begin streaming the response.
    pub fn submit_input(&mut self) {
        self.chat_pane.submit_input();
    }

    pub fn tick(&mut self) {
        let sample = self.live_monitor.sample();

        let ram_pct = sample.ram_used_bytes as f64 / sample.ram_total_bytes as f64;
        push_history(&mut self.ram_history, ram_pct, HISTORY_LEN);
        push_history(
            &mut self.cpu_history,
            sample.cpu_usage_percent / 100.0,
            HISTORY_LEN,
        );

        for (i, gpu) in sample.gpus.iter().enumerate() {
            if let Some(busy) = gpu.busy_percent {
                push_history(&mut self.gpu_history[i], busy / 100.0, HISTORY_LEN);
            }
            if let (Some(used), Some(total)) = (gpu.vram_used_bytes, gpu.vram_total_bytes) {
                let pct = used as f64 / total as f64;
                push_history(&mut self.vram_history[i], pct, HISTORY_LEN);
            }
        }
        self.latest_sample = Some(sample);

        // Drain streaming events for the chat pane.
        self.chat_pane.tick();
    }

    pub fn toggle_hw_panel(&mut self) {
        self.show_hw_panel = !self.show_hw_panel;
    }

    pub fn quit(&mut self) {
        self.should_quit = true;
    }
}

fn push_history(
    buf: &mut std::collections::VecDeque<f64>,
    val: f64,
    max: usize,
) {
    if buf.len() >= max {
        buf.pop_front();
    }
    buf.push_back(val);
}
