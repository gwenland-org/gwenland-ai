use crate::app::App;
use ratatui::{
    layout::{Constraint, Direction, Layout, Alignment},
    style::{Color, Modifier, Style, Stylize},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Chart, Dataset, Axis, GraphType},
    symbols::Marker,
    Frame,
};
use std::collections::VecDeque;

// --- Color Constants ---
const ORANGE: Color = Color::Rgb(255, 140, 66);
const BG: Color = Color::Rgb(18, 14, 28);
const BORDER: Color = Color::Rgb(38, 30, 58);
const DIM: Color = Color::Rgb(80, 70, 100);
const DIM_WHITE: Color = Color::Rgb(180, 180, 180);
const WHITE: Color = Color::Rgb(220, 220, 220);

// --- Style Constants (cached to avoid recreating every frame) ---
const STYLE_ORANGE_BOLD: Style = Style::new().fg(ORANGE).add_modifier(Modifier::BOLD);
const STYLE_ORANGE_BG: Style = Style::new().bg(ORANGE).fg(BG).add_modifier(Modifier::BOLD);
const STYLE_ORANGE_BG_PLAIN: Style = Style::new().bg(ORANGE).fg(BG);
const STYLE_DIM: Style = Style::new().fg(DIM);
const STYLE_DIM_BOLD: Style = Style::new().fg(DIM).add_modifier(Modifier::BOLD);
const STYLE_DIM_DIM: Style = Style::new().fg(DIM).add_modifier(Modifier::DIM);
const STYLE_WHITE: Style = Style::new().fg(WHITE);
const STYLE_BG: Style = Style::new().bg(BG);
const STYLE_BORDER: Style = Style::new().fg(BORDER);
const STYLE_ERROR: Style = Style::new().fg(Color::Rgb(120, 60, 60));
const STYLE_BADGE_EMPTY: Style = Style::new().fg(DIM);

pub fn render(app: &App, frame: &mut Frame) {
    // 1. Define vertical layouts for the 4 core sections
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // Top Bar
            Constraint::Min(0),    // Main Pane (centered splash or chat history)
            Constraint::Length(3), // Input Bar (Bordered block, enclosing 1 line content)
            Constraint::Length(1), // Keybind Bar
        ])
        .split(frame.area());

    // --- 1. TOP BAR ---
    let top_bar_bg = Paragraph::new("").style(Style::default().bg(ORANGE));
    frame.render_widget(top_bar_bg, chunks[0]);

    let top_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(0), Constraint::Length(20)])
        .split(chunks[0]);

    let top_left = Paragraph::new("✦ GWENCLI 2.0 — THE SENTINEL").style(STYLE_ORANGE_BG);
    frame.render_widget(top_left, top_layout[0]);

    let top_right = Paragraph::new("v2.0.0-alpha")
        .alignment(Alignment::Right)
        .style(STYLE_ORANGE_BG_PLAIN);
    frame.render_widget(top_right, top_layout[1]);

    // --- 2. MAIN PANE ---
    // If input is empty, render the centered ASCII art logo splash screen.
    // If input has >= 1 char, render active session chat logs.
    if app.chat_pane.input.is_empty() && app.chat_pane.messages.is_empty() {
        // Vertical layout inside the Main Pane to center the ASCII logo
        let main_height = chunks[1].height;
        let logo_lines_count = 9; // 6 lines logo + 1 blank line + 2 lines subtitles
        let top_padding = if main_height > logo_lines_count {
            (main_height - logo_lines_count) / 2
        } else {
            0
        };

        let center_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(top_padding),
                Constraint::Length(logo_lines_count),
                Constraint::Min(0),
            ])
            .split(chunks[1]);

        let ascii_logo = vec![
            Line::from(Span::styled("  ██████╗ ██╗    ██╗███████╗███╗   ██╗", STYLE_ORANGE_BOLD)),
            Line::from(Span::styled(" ██╔════╝ ██║    ██║██╔════╝████╗  ██║", STYLE_ORANGE_BOLD)),
            Line::from(Span::styled(" ██║  ███╗██║ █╗ ██║█████╗  ██╔██╗ ██║", STYLE_ORANGE_BOLD)),
            Line::from(Span::styled(" ██║   ██║██║███╗██║██╔══╝  ██║╚██╗██║", STYLE_ORANGE_BOLD)),
            Line::from(Span::styled(" ╚██████╔╝╚███╔███╔╝███████╗██║ ╚████║", STYLE_ORANGE_BOLD)),
            Line::from(Span::styled("  ╚═════╝  ╚══╝╚══╝ ╚══════╝╚═╝  ╚═══╝", STYLE_ORANGE_BOLD)),
            Line::from(Span::raw("")),
            Line::from(Span::styled("The Sentinel · v2.0.0-alpha", STYLE_DIM)),
            Line::from(Span::styled("start typing to begin", STYLE_DIM_DIM)),
        ];

        let logo_p = Paragraph::new(ascii_logo)
            .alignment(Alignment::Center)
            .style(STYLE_BG);
        frame.render_widget(logo_p, center_layout[1]);

        frame.render_widget(Block::default().bg(BG), center_layout[0]);
        frame.render_widget(Block::default().bg(BG), center_layout[2]);
    } else {
        // Active State: Render top info row and empty chat history below
        // Split Main Pane vertically into info row and chat area
        // Bento grid height: collapsed = 1, expanded = BENTO_CELL_H rows of cells
        // Each bento cell is BENTO_CELL_H lines tall (border + 1 stat line + 3 chart lines + border)
        const BENTO_CELL_H: u16 = 6;
        // GPU cells are taller: busy row + busy chart + vram row + vram chart + 2 borders
        const GPU_CELL_H: u16 = 10;

        let num_gpus = app.hardware.gpus.len();
        // Info row: bento grid top row (RAM+CPU+GPUs) + model panel beneath it, or collapsed
        // We stack: [bento row | model panel] vertically; bento row height = max(BENTO_CELL_H, GPU_CELL_H)
        let bento_row_h = if num_gpus > 0 { GPU_CELL_H } else { BENTO_CELL_H };
        let info_height: u16 = if app.show_hw_panel {
            bento_row_h + BENTO_CELL_H // bento row + model panel (reuses BENTO_CELL_H)
        } else {
            1 + BENTO_CELL_H // collapsed hw bar + model panel
        };

        let main_layout = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(info_height),
                Constraint::Min(0),
            ])
            .split(chunks[1]);

        // info_row = [hw section (bento row or collapsed bar)] stacked above [model panel]
        let info_stack = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(if app.show_hw_panel { bento_row_h } else { 1 }),
                Constraint::Length(BENTO_CELL_H),
            ])
            .split(main_layout[0]);

        // --- HW section: either collapsed bar or bento grid row ---
        if !app.show_hw_panel {
            let collapsed = Paragraph::new(Line::from(vec![
                Span::styled("  hw hidden ", STYLE_DIM),
                Span::styled("Ctrl+H", STYLE_ORANGE_BOLD),
                Span::styled(" to show", STYLE_DIM),
            ])).bg(BG);
            frame.render_widget(collapsed, info_stack[0]);
        } else {
            // Build bento grid: RAM | CPU | GPU0 | GPU1 | … all equal-width columns
            let num_cells = 2 + num_gpus; // RAM + CPU + GPUs
            let bento_constraints: Vec<Constraint> = (0..num_cells)
                .map(|_| Constraint::Ratio(1, num_cells as u32))
                .collect();

            let bento_cols = Layout::default()
                .direction(Direction::Horizontal)
                .constraints(bento_constraints)
                .split(info_stack[0]);

            // --- RAM cell ---
            let ram_block = Block::default()
                .title(Span::styled(" RAM ", STYLE_ORANGE_BOLD))
                .borders(Borders::ALL)
                .border_style(STYLE_BORDER)
                .bg(BG);
            let ram_inner = ram_block.inner(bento_cols[0]);
            frame.render_widget(ram_block, bento_cols[0]);

            let ram_cell_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1), Constraint::Min(0)])
                .split(ram_inner);

            let (used_gb, total_gb, ram_pct) = if let Some(ref sample) = app.latest_sample {
                let used = sample.ram_used_bytes as f64 / 1073741824.0;
                let total = sample.ram_total_bytes as f64 / 1073741824.0;
                let pct = if sample.ram_total_bytes > 0 {
                    (sample.ram_used_bytes as f64 / sample.ram_total_bytes as f64) * 100.0
                } else { 0.0 };
                (used, total, pct)
            } else {
                let used = app.hardware.total_ram_gb - app.hardware.available_ram_gb;
                let total = app.hardware.total_ram_gb;
                let pct = if total > 0.0 { (used / total) * 100.0 } else { 0.0 };
                (used, total, pct)
            };
            let ram_stat = Paragraph::new(Line::from(vec![
                Span::styled(format!(" {:.1}/{:.1}GB  {:.0}%", used_gb, total_gb, ram_pct), STYLE_WHITE),
            ])).bg(BG);
            frame.render_widget(ram_stat, ram_cell_layout[0]);

            let ram_data = history_to_dataset(&app.ram_history);
            let ram_dataset = Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(ORANGE))
                .data(&ram_data);
            let ram_chart = Chart::new(vec![ram_dataset])
                .x_axis(Axis::default().bounds([0.0, 59.0]))
                .y_axis(Axis::default().bounds([0.0, 1.0]))
                .bg(BG);
            frame.render_widget(ram_chart, ram_cell_layout[1]);

            // --- CPU cell ---
            let cpu_block = Block::default()
                .title(Span::styled(" CPU ", STYLE_ORANGE_BOLD))
                .borders(Borders::ALL)
                .border_style(STYLE_BORDER)
                .bg(BG);
            let cpu_inner = cpu_block.inner(bento_cols[1]);
            frame.render_widget(cpu_block, bento_cols[1]);

            let cpu_cell_layout = Layout::default()
                .direction(Direction::Vertical)
                .constraints([Constraint::Length(1), Constraint::Min(0)])
                .split(cpu_inner);

            let cpu_pct = app.latest_sample.as_ref().map(|s| s.cpu_usage_percent).unwrap_or(0.0);
            let cpu_short = if app.hardware.cpu_brand.contains('@') {
                let parts: Vec<&str> = app.hardware.cpu_brand.split('@').collect();
                let last = parts[0].trim().split_whitespace().next_back().unwrap_or("");
                format!("{}  {:.0}%", last, cpu_pct)
            } else {
                format!("{}c  {:.0}%", app.hardware.cpu_count, cpu_pct)
            };
            let cpu_stat = Paragraph::new(Line::from(vec![
                Span::styled(format!(" {}", cpu_short), STYLE_WHITE),
            ])).bg(BG);
            frame.render_widget(cpu_stat, cpu_cell_layout[0]);

            let cpu_data = history_to_dataset(&app.cpu_history);
            let cpu_dataset = Dataset::default()
                .marker(Marker::Braille)
                .graph_type(GraphType::Line)
                .style(Style::default().fg(ORANGE))
                .data(&cpu_data);
            let cpu_chart = Chart::new(vec![cpu_dataset])
                .x_axis(Axis::default().bounds([0.0, 59.0]))
                .y_axis(Axis::default().bounds([0.0, 1.0]))
                .bg(BG);
            frame.render_widget(cpu_chart, cpu_cell_layout[1]);

            // --- GPU cells ---
            for n in 0..num_gpus {
                let gpu = &app.hardware.gpus[n];
                let gpu_block = Block::default()
                    .title(Span::styled(format!(" GPU {} ", n), STYLE_ORANGE_BOLD))
                    .title_bottom(Span::styled(
                        format!(" {} ", gpu.name.chars().take(12).collect::<String>()),
                        STYLE_DIM,
                    ))
                    .borders(Borders::ALL)
                    .border_style(STYLE_BORDER)
                    .bg(BG);
                let gpu_inner = gpu_block.inner(bento_cols[2 + n]);
                frame.render_widget(gpu_block, bento_cols[2 + n]);

                let gpu_cell_layout = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(1), // busy stat
                        Constraint::Min(0),    // busy sparkline (fills remaining)
                        Constraint::Length(1), // vram stat
                        Constraint::Length(2), // vram sparkline (compact)
                    ])
                    .split(gpu_inner);

                let latest_gpu = app.latest_sample.as_ref().and_then(|s| s.gpus.get(n));
                let busy_val = latest_gpu.and_then(|g| g.busy_percent);
                let busy_text = busy_val.map(|b| format!(" busy  {:.0}%", b))
                    .unwrap_or_else(|| " busy  —".to_string());
                frame.render_widget(
                    Paragraph::new(Line::from(Span::styled(busy_text, STYLE_WHITE))).bg(BG),
                    gpu_cell_layout[0],
                );

                if busy_val.is_some() {
                    let gpu_busy_data = history_to_dataset(&app.gpu_history[n]);
                    let gpu_dataset = Dataset::default()
                        .marker(Marker::Braille)
                        .graph_type(GraphType::Line)
                        .style(Style::default().fg(ORANGE))
                        .data(&gpu_busy_data);
                    let gpu_chart = Chart::new(vec![gpu_dataset])
                        .x_axis(Axis::default().bounds([0.0, 59.0]))
                        .y_axis(Axis::default().bounds([0.0, 1.0]))
                        .bg(BG);
                    frame.render_widget(gpu_chart, gpu_cell_layout[1]);
                } else {
                    frame.render_widget(Paragraph::new("").bg(BG), gpu_cell_layout[1]);
                }

                let vram_used = latest_gpu.and_then(|g| g.vram_used_bytes);
                let vram_total = latest_gpu.and_then(|g| g.vram_total_bytes);
                let vram_text = if let (Some(used), Some(total)) = (vram_used, vram_total) {
                    let pct = if total > 0 { (used as f64 / total as f64) * 100.0 } else { 0.0 };
                    format!(" vram  {} / {}  {:.0}%",
                        format_bytes_to_mb_or_gb(used),
                        format_bytes_to_mb_or_gb(total),
                        pct)
                } else {
                    " vram  —".to_string()
                };
                frame.render_widget(
                    Paragraph::new(Line::from(Span::styled(vram_text, STYLE_WHITE))).bg(BG),
                    gpu_cell_layout[2],
                );

                if vram_used.is_some() && vram_total.is_some() {
                    let vram_data = history_to_dataset(&app.vram_history[n]);
                    let vram_dataset = Dataset::default()
                        .marker(Marker::Braille)
                        .graph_type(GraphType::Line)
                        .style(Style::default().fg(ORANGE))
                        .data(&vram_data);
                    let vram_chart = Chart::new(vec![vram_dataset])
                        .x_axis(Axis::default().bounds([0.0, 59.0]))
                        .y_axis(Axis::default().bounds([0.0, 1.0]))
                        .bg(BG);
                    frame.render_widget(vram_chart, gpu_cell_layout[3]);
                } else {
                    frame.render_widget(Paragraph::new("").bg(BG), gpu_cell_layout[3]);
                }
            }
        }

        // --- Model Panel (below bento grid, full width) ---
        let model_name = app.active_model.as_deref().unwrap_or("no model loaded");
        let model_name_line = Line::from(vec![
            Span::styled("name          ", STYLE_DIM),
            Span::styled(model_name, if app.active_model.is_some() { STYLE_WHITE } else { STYLE_DIM }),
        ]);
        let model_provider_line = Line::from(vec![
            Span::styled("provider      ", STYLE_DIM),
            Span::styled(if app.active_model.is_some() { "local" } else { "—" }, if app.active_model.is_some() { STYLE_WHITE } else { STYLE_DIM }),
        ]);
        let model_quantize_line = Line::from(vec![
            Span::styled("quantize      ", STYLE_DIM),
            Span::styled("—", STYLE_DIM),
        ]);
        let model_status_line = Line::from(vec![
            Span::styled("status        ", STYLE_DIM),
            if app.active_model.is_some() {
                Span::styled("● ready", Style::new().fg(Color::Rgb(60, 180, 60)))
            } else {
                Span::styled("● not ready", STYLE_ERROR)
            },
        ]);

        let model_block = Block::default()
            .title(Span::styled("  Active Model ", STYLE_ORANGE_BOLD))
            .borders(Borders::ALL)
            .border_style(STYLE_BORDER)
            .bg(BG);

        let model_content = vec![
            model_name_line,
            model_provider_line,
            model_quantize_line,
            model_status_line,
        ];
        let model_paragraph = Paragraph::new(model_content)
            .block(model_block);

        frame.render_widget(model_paragraph, info_stack[1]);

        // 3. Chat Area — delegated to ChatPane which owns all message/streaming state
        app.chat_pane.render(frame, main_layout[1]);
    }

    // --- 3. INPUT BAR ---
    let input_block = Block::default()
        .borders(Borders::ALL)
        .border_style(STYLE_BORDER)
        .bg(BG);
    let inner_area = input_block.inner(chunks[2]);
    frame.render_widget(input_block, chunks[2]);

    let input_layout = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(40),
        ])
        .split(inner_area);

    let typed_text = Line::from(vec![
        Span::styled("› ", STYLE_ORANGE_BOLD),
        Span::raw(&app.chat_pane.input),
    ]);
    let input_p = Paragraph::new(typed_text);
    frame.render_widget(input_p, input_layout[0]);

    let token_lock = app.token_estimate.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let model_str = if let Some(ref model) = app.active_model {
        format!("[ {} ]", model)
    } else {
        "[ no model loaded ]".to_string()
    };

    let badge_text = if let Some(ref model_name) = app.active_model {
        if let Some(ref estimate) = *token_lock {
            // BOTH model is loaded AND scan is complete
            let status_color = match estimate.status {
                gwenland_core::tokenizer::TokenStatus::Safe => Color::Green,
                gwenland_core::tokenizer::TokenStatus::Warning => Color::Yellow,
                gwenland_core::tokenizer::TokenStatus::Critical | gwenland_core::tokenizer::TokenStatus::Exceeded => Color::Red,
            };
            
            let token_str = format!(" {} / {} tokens ", estimate.total_tokens, estimate.budget);
            
            vec![
                Span::styled(token_str, Style::new().fg(status_color)),
                Span::styled(" ", Style::new()),
                Span::styled(model_str, STYLE_WHITE),
            ]
        } else {
            // Model IS loaded but token scan still in progress
            let budget = gwenland_core::tokenizer::detect_budget_from_model(model_name);
            let budget_val = match budget {
                gwenland_core::tokenizer::ModelBudget::Small => gwenland_core::tokenizer::BUDGET_8K,
                gwenland_core::tokenizer::ModelBudget::Medium => gwenland_core::tokenizer::BUDGET_32K,
                gwenland_core::tokenizer::ModelBudget::Large => gwenland_core::tokenizer::BUDGET_128K,
                gwenland_core::tokenizer::ModelBudget::Custom(val) => val,
            };
            let token_str = format!(" ... / {} tokens ", budget_val);
            
            vec![
                Span::styled(token_str, STYLE_BADGE_EMPTY),
                Span::styled(" ", Style::new()),
                Span::styled(model_str, STYLE_BADGE_EMPTY),
            ]
        }
    } else {
        // `active_model` is None/missing from config
        vec![
            Span::styled(" -- / -- tokens ", STYLE_BADGE_EMPTY),
            Span::styled(" ", Style::new()),
            Span::styled(model_str, STYLE_BADGE_EMPTY),
        ]
    };
    let badge_p = Paragraph::new(Line::from(badge_text))
        .alignment(Alignment::Right);
    frame.render_widget(badge_p, input_layout[1]);

    // Set cursor position: at the end of input text
    // inner_area.x + 2 (length of "› ") + app.input.len()
    let cursor_x = inner_area.x + 2 + app.chat_pane.input.len() as u16;
    let cursor_y = inner_area.y;
    frame.set_cursor_position((cursor_x, cursor_y));

    // --- 4. KEYBIND BAR ---
    let keybind_block = Block::default().bg(BG);
    let keybinds_text = Line::from(vec![
        Span::styled(" M ", STYLE_ORANGE_BOLD),
        Span::styled("Models", STYLE_DIM),
        Span::styled("  |  ", STYLE_BORDER),
        Span::styled(" C ", STYLE_ORANGE_BOLD),
        Span::styled("Context", STYLE_DIM),
        Span::styled("  |  ", STYLE_BORDER),
        Span::styled(" G ", STYLE_ORANGE_BOLD),
        Span::styled("GUI", STYLE_DIM),
        Span::styled("  |  ", STYLE_BORDER),
        Span::styled(" Ctrl+H ", STYLE_ORANGE_BOLD),
        Span::styled("HW", STYLE_DIM),
        Span::styled("  |  ", STYLE_BORDER),
        Span::styled(" H ", STYLE_ORANGE_BOLD),
        Span::styled("Help", STYLE_DIM),
        Span::styled("  |  ", STYLE_BORDER),
        Span::styled(" Q ", STYLE_ORANGE_BOLD),
        Span::styled("Quit", STYLE_DIM),
    ]);
    let keybind_p = Paragraph::new(keybinds_text)
        .block(keybind_block);
    frame.render_widget(keybind_p, chunks[3]);
}

fn history_to_dataset(history: &VecDeque<f64>) -> Vec<(f64, f64)> {
    history.iter().enumerate()
        .map(|(i, &v)| (i as f64, v))
        .collect()
}

fn format_bytes_to_mb_or_gb(bytes: u64) -> String {
    let mb = bytes as f64 / 1024.0 / 1024.0;
    if mb >= 1024.0 {
        format!("{:.1} GB", mb / 1024.0)
    } else {
        format!("{:.0} MB", mb)
    }
}
