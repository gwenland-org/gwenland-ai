use ratatui::{layout::Rect, Frame};
use crossterm::event::{KeyEvent, MouseEvent};
use crate::app::PaneId;

pub mod chat;
pub mod train;
pub mod fetch;
pub mod engines;
pub mod benchmark;

pub enum PaneAction {
    None,
    SwitchPane(PaneId),
}

pub trait Pane {
    fn draw(&mut self, f: &mut Frame, area: Rect);
    fn handle_key(&mut self, key: KeyEvent) -> PaneAction { PaneAction::None }
    fn handle_mouse(&mut self, mouse: MouseEvent) -> PaneAction { PaneAction::None }
    fn tick(&mut self) {}
}
