use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::{Block, BorderType, Borders, Paragraph},
    Frame,
};
use crossterm::event::{KeyEvent, MouseEvent};
use tui_textarea::TextArea;
use crate::ui::theme;
use super::{Pane, PaneAction};
use crate::ui::slash_popup::{SlashPopup, SlashAction};

/// Truncate a string to `max` characters, appending an ellipsis if cut.
fn truncate(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        s.to_string()
    } else {
        let keep = max.saturating_sub(1);
        format!("{}…", s.chars().take(keep).collect::<String>())
    }
}

pub struct ChatPane<'a> {
    textarea: TextArea<'a>,
    history: Vec<String>,
    popup: SlashPopup<'a>,
    /// Device summary rows (label, value) shown on the welcome card. Detected
    /// once at construction so GPU/RAM probing never runs on the render path.
    device_info: Vec<(String, String)>,
}

impl<'a> ChatPane<'a> {
    pub fn new() -> Self {
        let textarea = Self::styled_textarea();
        Self {
            textarea,
            history: Vec::new(),
            popup: SlashPopup::new(),
            device_info: Self::detect_device_info(),
        }
    }

    /// Probe host hardware via gwenland-core and format it into compact
    /// label/value rows for the welcome card. Runs once at startup.
    fn detect_device_info() -> Vec<(String, String)> {
        use gwenland_core::platform::hardware::{self, Arch, GpuType};

        let p = hardware::profile();

        let arch = match p.arch {
            Arch::X86_64 => "x86_64",
            Arch::Aarch64 => "aarch64",
            Arch::Unknown => std::env::consts::ARCH,
        };

        // CPU brand can be long ("11th Gen Intel(R) Core(TM) i3-1115G4 @ ...");
        // strip vendor noise and the clock suffix so the row fits the card.
        let cpu = {
            let brand = p.cpu_brand.trim();
            let short = brand.split('@').next().unwrap_or(brand);
            let short = short
                .replace("(R)", "")
                .replace("(TM)", "")
                .replace("CPU", "");
            let short = short.split_whitespace().collect::<Vec<_>>().join(" ");
            format!("{}  ·  {} cores", truncate(&short, 26), p.cpu_count)
        };

        let mut rows = vec![
            ("CPU".to_string(), cpu),
            (
                "RAM".to_string(),
                format!(
                    "{:.1} / {:.1} GB free",
                    p.available_ram_gb, p.total_ram_gb
                ),
            ),
            ("Arch".to_string(), format!("{}  ·  {}", arch, std::env::consts::OS)),
        ];

        // Prefer a dedicated GPU, else the first detected adapter.
        let gpu = p
            .gpus
            .iter()
            .find(|g| g.gpu_type == GpuType::Dedicated)
            .or_else(|| p.gpus.first());
        if let Some(g) = gpu {
            let kind = match g.gpu_type {
                GpuType::Dedicated => "dedicated",
                GpuType::Integrated => "integrated",
                GpuType::Unknown => "gpu",
            };
            let gpu_name = g.name.replace("(R)", "").replace("(TM)", "");
            let gpu_name = gpu_name.split_whitespace().collect::<Vec<_>>().join(" ");
            rows.push(("GPU".to_string(), format!("{}  ·  {}", truncate(&gpu_name, 26), kind)));
        }

        rows
    }

    /// A fresh composer textarea styled for the input box (placeholder + colors).
    fn styled_textarea() -> TextArea<'a> {
        let mut textarea = TextArea::default();
        textarea.set_style(Style::default().bg(theme::INPUT_BG).fg(theme::TEXT));
        textarea.set_cursor_line_style(Style::default().bg(theme::INPUT_BG));
        textarea.set_placeholder_text("Type a message or / for commands");
        textarea.set_placeholder_style(Style::default().fg(theme::TEXT_DIM).bg(theme::INPUT_BG));
        textarea
    }
}

impl<'a> Pane for ChatPane<'a> {
    fn draw(&mut self, f: &mut Frame, area: Rect) {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),    // History
                Constraint::Length(1), // Spacer
                Constraint::Length(3), // Input area (height 3 for border padding)
                Constraint::Length(1), // Status row
            ])
            .margin(2) // global margin
            .split(area);

        // Draw background for whole pane
        f.render_widget(Block::default().style(Style::default().bg(theme::BG)), area);

        if self.history.is_empty() {
             // Card = header (title + subtitle) + separator + device rows + hint.
             // Height grows with the number of device rows detected.
             let device_rows = self.device_info.len() as u16;
             let box_height: u16 = 4 + 1 + device_rows + 2; // header(3)+pad+sep+rows+pad+hint
             let v = Layout::default()
                 .direction(Direction::Vertical)
                 .constraints([
                     Constraint::Min(0),
                     Constraint::Length(box_height),
                     Constraint::Min(0),
                 ])
                 .split(chunks[0]);

             // Horizontally center the card at a fixed comfortable width.
             let box_width: u16 = 52;
             let h = Layout::default()
                 .direction(Direction::Horizontal)
                 .constraints([
                     Constraint::Min(0),
                     Constraint::Length(box_width),
                     Constraint::Min(0),
                 ])
                 .split(v[1]);

             let card = Block::default()
                 .borders(Borders::ALL)
                 .border_type(BorderType::Rounded)
                 .border_style(Style::default().fg(theme::BORDER))
                 .style(Style::default().bg(theme::BG))
                 .padding(ratatui::widgets::Padding::horizontal(2));
             let inner = card.inner(h[1]);
             f.render_widget(card, h[1]);

             let mut text = vec![
                 Line::from(""),
                 Line::from(Span::styled(
                     "GwenLand",
                     Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD),
                 ))
                 .alignment(Alignment::Center),
                 Line::from(Span::styled(
                     "Local AI. Your machine.",
                     Style::default().fg(theme::TEXT_SECONDARY),
                 ))
                 .alignment(Alignment::Center),
             ];

             // Separator rule between the header and the device block.
             let rule_w = inner.width as usize;
             text.push(
                 Line::from(Span::styled(
                     "─".repeat(rule_w),
                     Style::default().fg(theme::BORDER),
                 )),
             );

             // Device rows: dim label on the left, value on the right, padded so
             // values line up in a column.
             let label_w = self
                 .device_info
                 .iter()
                 .map(|(l, _)| l.chars().count())
                 .max()
                 .unwrap_or(0);
             for (label, value) in &self.device_info {
                 let pad = label_w.saturating_sub(label.chars().count());
                 text.push(Line::from(vec![
                     Span::styled(
                         format!("{}{}  ", label, " ".repeat(pad)),
                         Style::default().fg(theme::TEXT_DIM),
                     ),
                     Span::styled(value.clone(), Style::default().fg(theme::TEXT_SECONDARY)),
                 ]));
             }

             text.push(Line::from(""));
             text.push(
                 Line::from(vec![
                     Span::styled("/ ", Style::default().fg(theme::ACCENT)),
                     Span::styled("commands    ", Style::default().fg(theme::TEXT_DIM)),
                     Span::styled("ctrl+c ", Style::default().fg(theme::ACCENT)),
                     Span::styled("chat", Style::default().fg(theme::TEXT_DIM)),
                 ])
                 .alignment(Alignment::Center),
             );

             f.render_widget(Paragraph::new(text), inner);
        } else {
            let history_text = self.history.join("\n\n");
            let history_widget = Paragraph::new(history_text)
                .style(Style::default().fg(theme::TEXT).bg(theme::BG));
            f.render_widget(history_widget, chunks[0]);
        }
        
        let composer_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(100),
            ])
            .split(chunks[2]);

        // Input is "focused" whenever the slash popup is not capturing keys.
        let input_focused = !self.popup.active;
        let border_color = if input_focused { theme::BORDER_ACTIVE } else { theme::BORDER };

        let input_block = Block::default()
            .borders(Borders::ALL)
            .border_type(BorderType::Rounded)
            .border_style(Style::default().fg(border_color))
            .padding(ratatui::widgets::Padding::horizontal(1))
            .style(Style::default().bg(theme::INPUT_BG));

        let inner_area = input_block.inner(composer_layout[0]);
        f.render_widget(input_block, composer_layout[0]);

        let prompt_chunk = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(2), Constraint::Min(0)])
            .split(inner_area);

        f.render_widget(Paragraph::new("> ").style(Style::default().fg(theme::ACCENT).bg(theme::INPUT_BG)), prompt_chunk[0]);
        f.render_widget(&self.textarea, prompt_chunk[1]);
        
        // Subtle single hint below the composer. Branding/version now lives in
        // the global status bar, so this row stays quiet.
        let hint = Line::from(vec![
            Span::styled("enter ", Style::default().fg(theme::TEXT_SECONDARY)),
            Span::styled("send   ", Style::default().fg(theme::TEXT_DIM)),
            Span::styled("esc ", Style::default().fg(theme::TEXT_SECONDARY)),
            Span::styled("close", Style::default().fg(theme::TEXT_DIM)),
        ]);
        f.render_widget(Paragraph::new(hint), chunks[3]);

        if self.popup.active {
            // Float the popup just above the composer, clamped so it never runs
            // off the top of the pane.
            let composer = composer_layout[0];
            let want = self.popup.desired_height();
            let available = composer.y.saturating_sub(area.y);
            let popup_height = want.min(available).max(3);
            let popup_area = Rect {
                x: composer.x,
                y: composer.y.saturating_sub(popup_height),
                width: composer.width,
                height: popup_height,
            };
            self.popup.draw(f, popup_area);
        }
    }

    fn handle_key(&mut self, key: KeyEvent) -> PaneAction {
        if self.popup.active {
            let action = self.popup.handle_key(key, &mut String::new());
            
            match action {
                SlashAction::Close => {
                    self.popup.close();
                    self.textarea = Self::styled_textarea();
                    return PaneAction::None;
                }
                SlashAction::SwitchPane(p) => {
                    self.popup.close();
                    self.textarea = Self::styled_textarea();
                    return PaneAction::SwitchPane(p);
                }
                SlashAction::ClearHistory => {
                    self.history.clear();
                    self.popup.close();
                    self.textarea = Self::styled_textarea();
                    return PaneAction::None;
                }
                SlashAction::Handled => {
                    return PaneAction::None;
                }
                SlashAction::None => {
                    // Let TextArea handle character input
                    self.textarea.input(key);
                    let query = self.textarea.lines().join("");
                    if query.is_empty() {
                        self.popup.close();
                    } else {
                        self.popup.update_filter(&query);
                    }
                    return PaneAction::None;
                }
            }
        }

        match key.code {
            crossterm::event::KeyCode::Enter => {
                let text = self.textarea.lines().join("\n");
                if !text.is_empty() {
                    self.history.push(format!("User: {}\n\nModel: ...", text));
                    self.textarea = Self::styled_textarea();
                }
                PaneAction::None
            }
            _ => {
                self.textarea.input(key);
                let text = self.textarea.lines().join("");
                if text.starts_with('/') {
                    self.popup.open();
                    self.popup.update_filter(&text);
                }
                PaneAction::None
            }
        }
    }

    fn handle_mouse(&mut self, mouse: MouseEvent) -> PaneAction {
        if !self.popup.active {
            self.textarea.input(crossterm::event::Event::Mouse(mouse));
        }
        PaneAction::None
    }

    fn tick(&mut self) {}
}
