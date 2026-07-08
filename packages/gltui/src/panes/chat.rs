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

pub struct ChatPane<'a> {
    textarea: TextArea<'a>,
    history: Vec<String>,
    popup: SlashPopup<'a>,
}

impl<'a> ChatPane<'a> {
    pub fn new() -> Self {
        let textarea = Self::styled_textarea();
        Self {
            textarea,
            history: Vec::new(),
            popup: SlashPopup::new(),
        }
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
             // Vertically center a compact welcome card in the history area.
             let box_height: u16 = 8;
             let v = Layout::default()
                 .direction(Direction::Vertical)
                 .constraints([
                     Constraint::Min(0),
                     Constraint::Length(box_height),
                     Constraint::Min(0),
                 ])
                 .split(chunks[0]);

             // Horizontally center the card at a fixed comfortable width.
             let box_width: u16 = 44;
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
                 .style(Style::default().bg(theme::BG));
             let inner = card.inner(h[1]);
             f.render_widget(card, h[1]);

             let text = vec![
                 Line::from(""),
                 Line::from(Span::styled(
                     "GwenLand",
                     Style::default().fg(theme::TEXT).add_modifier(Modifier::BOLD),
                 )),
                 Line::from(Span::styled(
                     "Local AI. Your machine.",
                     Style::default().fg(theme::TEXT_SECONDARY),
                 )),
                 Line::from(""),
                 Line::from(vec![
                     Span::styled("/ ", Style::default().fg(theme::ACCENT)),
                     Span::styled("commands    ", Style::default().fg(theme::TEXT_DIM)),
                     Span::styled("ctrl+c ", Style::default().fg(theme::ACCENT)),
                     Span::styled("chat", Style::default().fg(theme::TEXT_DIM)),
                 ]),
             ];

             f.render_widget(
                 Paragraph::new(text).alignment(Alignment::Center),
                 inner,
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
