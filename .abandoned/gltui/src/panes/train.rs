use ratatui::{
    layout::Rect,
    widgets::{Block, Borders, Paragraph},
    Frame,
};
use crossterm::event::KeyEvent;
use crate::ui::theme;
use super::Pane;

pub struct TrainPane;

impl TrainPane {
    pub fn new() -> Self {
        Self
    }
}

impl Pane for TrainPane {
    fn draw(&mut self, f: &mut Frame, area: Rect) {
        let p = Paragraph::new("Training monitor placeholder")
            .style(ratatui::style::Style::default().fg(theme::TEXT).bg(theme::BG));
        f.render_widget(p, area);
    }

    fn handle_key(&mut self, _key: KeyEvent) -> crate::panes::PaneAction {
        crate::panes::PaneAction::None
    }
}
