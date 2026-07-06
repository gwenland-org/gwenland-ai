use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::Style,
    widgets::{Block, Borders, Paragraph},
    Frame,
};
use crossterm::event::{KeyEvent, MouseEvent};
use tui_textarea::TextArea;
use crate::ui::theme;
use super::{Pane, PaneAction};
use crate::ui::slash_popup::{SlashPopup, SlashAction};

pub struct ChatPane<'a> {
    textarea: TextArea<'a>,
    history: Vec<String>,
    popup: SlashPopup<'a>,
}

impl<'a> ChatPane<'a> {
    pub fn new() -> Self {
        let mut textarea = TextArea::default();
        textarea.set_style(Style::default().bg(theme::SURFACE).fg(theme::TEXT));
        textarea.set_cursor_line_style(Style::default().bg(theme::SURFACE));
        Self {
            textarea,
            history: Vec::new(),
            popup: SlashPopup::new(),
        }
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
             let welcome_layout = Layout::default()
                 .direction(Direction::Vertical)
                 .constraints([
                     Constraint::Min(0),
                     Constraint::Length(12),
                     Constraint::Min(0),
                 ])
                 .split(chunks[0]);
                 
             let logo = r#"   ____                     __                  __ 
  / ___|__      _____ _ __  \ \   __ _ _ __   __| |
 | |  _ \ \ /\ / / _ \ '_ \  | | / _` | '_ \ / _` |
 | |_| | \ V  V /  __/ | | | | || (_| | | | | (_| |
  \____|  \_/\_/ \___|_| |_| |___\__,_|_| |_|\__,_|"#;

             let mut text = vec![];
             for line in logo.lines() {
                 if !line.is_empty() {
                     text.push(ratatui::text::Line::from(ratatui::text::Span::styled(line, Style::default().fg(theme::PRIMARY).add_modifier(ratatui::style::Modifier::BOLD))));
                 }
             }
             text.push(ratatui::text::Line::from(""));
             text.push(ratatui::text::Line::from(ratatui::text::Span::styled("Welcome to GwenLand Core", Style::default().fg(theme::TEXT).add_modifier(ratatui::style::Modifier::BOLD))));
             text.push(ratatui::text::Line::from(""));
             text.push(ratatui::text::Line::from(ratatui::text::Span::styled("Type / for commands", Style::default().fg(theme::MUTED))));
             text.push(ratatui::text::Line::from(ratatui::text::Span::styled("Ctrl+T to Train Models", Style::default().fg(theme::MUTED))));
             text.push(ratatui::text::Line::from(ratatui::text::Span::styled("Ctrl+F to Fetch Models", Style::default().fg(theme::MUTED))));
             text.push(ratatui::text::Line::from(ratatui::text::Span::styled("Ctrl+E to manage Engines", Style::default().fg(theme::MUTED))));

             f.render_widget(
                 Paragraph::new(text).alignment(ratatui::layout::Alignment::Center),
                 welcome_layout[1]
             );
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
        
        let input_block = Block::default()
            .padding(ratatui::widgets::Padding::symmetric(2, 1))
            .style(Style::default().bg(theme::SURFACE));
            
        let inner_area = input_block.inner(composer_layout[0]);
        f.render_widget(input_block, composer_layout[0]);
        
        let prompt_chunk = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Length(2), Constraint::Min(0)])
            .split(inner_area);
            
        f.render_widget(Paragraph::new("> ").style(Style::default().fg(theme::CYAN).bg(theme::SURFACE)), prompt_chunk[0]);
        f.render_widget(&self.textarea, prompt_chunk[1]);
        
        let status_row_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([
                Constraint::Percentage(100),
            ])
            .split(chunks[3]);
            
        let status_layout = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(status_row_layout[0]);
            
        let left_status = ratatui::text::Line::from(vec![
            ratatui::text::Span::styled("enter ", Style::default().fg(theme::TEXT)),
            ratatui::text::Span::styled("send", Style::default().fg(theme::MUTED)),
        ]);
        f.render_widget(Paragraph::new(left_status), status_layout[0]);
        
        let right_status = ratatui::text::Line::from(vec![
            ratatui::text::Span::styled("GwenLand ", Style::default().fg(theme::MUTED)),
            ratatui::text::Span::styled("Core 1.0", Style::default().fg(theme::TEXT)),
        ]);
        f.render_widget(Paragraph::new(right_status).alignment(ratatui::layout::Alignment::Right), status_layout[1]);

        if self.popup.active {
            let popup_height = 12;
            let popup_area = Rect {
                x: composer_layout[0].x,
                y: composer_layout[0].y.saturating_sub(popup_height),
                width: composer_layout[0].width,
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
                    self.textarea = TextArea::default();
                    self.textarea.set_style(Style::default().bg(theme::SURFACE).fg(theme::TEXT));
                    self.textarea.set_cursor_line_style(Style::default().bg(theme::SURFACE));
                    return PaneAction::None;
                }
                SlashAction::SwitchPane(p) => {
                    self.popup.close();
                    self.textarea = TextArea::default();
                    self.textarea.set_style(Style::default().bg(theme::SURFACE).fg(theme::TEXT));
                    self.textarea.set_cursor_line_style(Style::default().bg(theme::SURFACE));
                    return PaneAction::SwitchPane(p);
                }
                SlashAction::ClearHistory => {
                    self.history.clear();
                    self.popup.close();
                    self.textarea = TextArea::default();
                    self.textarea.set_style(Style::default().bg(theme::SURFACE).fg(theme::TEXT));
                    self.textarea.set_cursor_line_style(Style::default().bg(theme::SURFACE));
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
                    self.textarea = TextArea::default();
                    self.textarea.set_style(Style::default().bg(theme::SURFACE).fg(theme::TEXT));
                    self.textarea.set_cursor_line_style(Style::default().bg(theme::SURFACE));
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
