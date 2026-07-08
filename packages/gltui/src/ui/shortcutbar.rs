use ratatui::{
    layout::{Alignment, Constraint, Direction, Layout, Rect},
    style::{Modifier, Style},
    text::{Line, Span},
    widgets::Paragraph,
    Frame,
};
use crate::app::PaneId;
use crate::ui::theme;

pub struct ShortcutBar;

impl ShortcutBar {
    pub fn draw(f: &mut Frame, area: Rect, active: PaneId) {
        // Full-width surface background for the whole bar.
        f.render_widget(
            Paragraph::new("").style(Style::default().bg(theme::SURFACE)),
            area,
        );

        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Min(0), Constraint::Length(20)])
            .split(area);

        // Left: tab-style mode indicators. Active mode is accent+bold, the rest
        // dim, separated by a dim bar. Then a dim "/ commands" hint.
        let modes = [
            ("chat", PaneId::Chat),
            ("train", PaneId::Train),
            ("fetch", PaneId::Fetch),
            ("engines", PaneId::Engines),
        ];

        let mut spans: Vec<Span> = vec![Span::styled(" ", Style::default().bg(theme::SURFACE))];
        for (i, (label, id)) in modes.iter().enumerate() {
            if i > 0 {
                spans.push(Span::styled(" │ ", Style::default().fg(theme::TEXT_DIM).bg(theme::SURFACE)));
            }
            let style = if *id == active {
                Style::default().fg(theme::ACCENT).bg(theme::SURFACE).add_modifier(Modifier::BOLD)
            } else {
                Style::default().fg(theme::TEXT_DIM).bg(theme::SURFACE)
            };
            spans.push(Span::styled(*label, style));
        }
        spans.push(Span::styled("    ", Style::default().bg(theme::SURFACE)));
        spans.push(Span::styled("/ ", Style::default().fg(theme::TEXT_DIM).bg(theme::SURFACE)));
        spans.push(Span::styled("commands", Style::default().fg(theme::TEXT_DIM).bg(theme::SURFACE)));

        f.render_widget(
            Paragraph::new(Line::from(spans)).style(Style::default().bg(theme::SURFACE)),
            cols[0],
        );

        // Right: product name (secondary) + version (dim), right-aligned.
        let right = Line::from(vec![
            Span::styled("GwenLand ", Style::default().fg(theme::TEXT_SECONDARY).bg(theme::SURFACE)),
            Span::styled("v1.0 ", Style::default().fg(theme::TEXT_DIM).bg(theme::SURFACE)),
        ]);
        f.render_widget(
            Paragraph::new(right)
                .alignment(Alignment::Right)
                .style(Style::default().bg(theme::SURFACE)),
            cols[1],
        );
    }
}
