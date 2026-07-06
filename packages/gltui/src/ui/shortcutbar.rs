use ratatui::{
    layout::Rect,
    style::{Color, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use crate::ui::theme;

pub struct ShortcutBar;

impl ShortcutBar {
    pub fn draw(f: &mut Frame, area: Rect) {
        let text = vec![
            Span::styled("ctrl+c ", Style::default().fg(theme::MUTED)),
            Span::styled("chat  ", Style::default().fg(theme::TEXT)),
            Span::styled("ctrl+t ", Style::default().fg(theme::MUTED)),
            Span::styled("train  ", Style::default().fg(theme::TEXT)),
            Span::styled("ctrl+f ", Style::default().fg(theme::MUTED)),
            Span::styled("fetch  ", Style::default().fg(theme::TEXT)),
            Span::styled("ctrl+e ", Style::default().fg(theme::MUTED)),
            Span::styled("engines  ", Style::default().fg(theme::TEXT)),
            Span::styled("/ ", Style::default().fg(theme::MUTED)),
            Span::styled("commands", Style::default().fg(theme::TEXT)),
        ];
        
        let p = Paragraph::new(Line::from(text))
            .style(Style::default().bg(theme::BG))
            .alignment(ratatui::layout::Alignment::Center);
        f.render_widget(p, area);
    }
}
