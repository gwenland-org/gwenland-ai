use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    Frame,
};
use crate::app::App;
use super::shortcutbar::ShortcutBar;

pub struct RootLayout;

impl RootLayout {
    pub fn draw(f: &mut Frame, app: &mut App) {
        let size = f.area();
        
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),
                Constraint::Length(1), // ShortcutBar
            ])
            .split(size);
        
        app.active_pane_mut().draw(f, chunks[0]);
        ShortcutBar::draw(f, chunks[1]);
    }
}
