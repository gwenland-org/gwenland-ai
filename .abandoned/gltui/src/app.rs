use std::io;
use ratatui::{backend::Backend, Terminal};
use crate::events::{Event, EventHandler};
use crate::ui::layout::RootLayout;
use crate::panes::{
    chat::ChatPane, train::TrainPane, fetch::FetchPane,
    engines::EnginesPane, benchmark::BenchmarkPane,
    Pane,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PaneId {
    Chat,
    Train,
    Fetch,
    Engines,
    Benchmark,
}

pub struct App {
    pub active_pane: PaneId,
    pub running: bool,
    
    chat_pane: ChatPane<'static>,
    train_pane: TrainPane,
    fetch_pane: FetchPane,
    engines_pane: EnginesPane,
    benchmark_pane: BenchmarkPane,
}

impl App {
    pub fn new(initial_pane: PaneId) -> Self {
        Self {
            active_pane: initial_pane,
            running: true,
            chat_pane: ChatPane::new(),
            train_pane: TrainPane::new(),
            fetch_pane: FetchPane::new(),
            engines_pane: EnginesPane::new(),
            benchmark_pane: BenchmarkPane::new(),
        }
    }

    pub async fn run<B: Backend>(&mut self, terminal: &mut Terminal<B>) -> io::Result<()> {
        let mut events = EventHandler::new(250);

        while self.running {
            terminal.draw(|f| {
                RootLayout::draw(f, self);
            })?;

            match events.next().await? {
                Event::Tick => {
                    self.active_pane_mut().tick();
                }
                Event::Key(key) => {
                    // Global hotkeys
                    match key.code {
                        crossterm::event::KeyCode::Char('q') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                            self.running = false;
                        }
                        crossterm::event::KeyCode::Char('c') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                            self.active_pane = PaneId::Chat;
                        }
                        crossterm::event::KeyCode::Char('t') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                            self.active_pane = PaneId::Train;
                        }
                        crossterm::event::KeyCode::Char('f') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                            self.active_pane = PaneId::Fetch;
                        }
                        crossterm::event::KeyCode::Char('e') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                            self.active_pane = PaneId::Engines;
                        }
                        crossterm::event::KeyCode::Char('b') if key.modifiers.contains(crossterm::event::KeyModifiers::CONTROL) => {
                            self.active_pane = PaneId::Benchmark;
                        }
                        _ => {
                            // Forward all other keys to the active pane
                            if let crate::panes::PaneAction::SwitchPane(p) = self.active_pane_mut().handle_key(key) {
                                self.active_pane = p;
                            }
                        }
                    }
                }
                Event::Mouse(mouse) => {
                    if let crate::panes::PaneAction::SwitchPane(p) = self.active_pane_mut().handle_mouse(mouse) {
                        self.active_pane = p;
                    }
                }
                Event::Resize(_, _) => {}
            }
        }
        Ok(())
    }

    pub fn active_pane_mut(&mut self) -> &mut dyn Pane {
        match self.active_pane {
            PaneId::Chat => &mut self.chat_pane,
            PaneId::Train => &mut self.train_pane,
            PaneId::Fetch => &mut self.fetch_pane,
            PaneId::Engines => &mut self.engines_pane,
            PaneId::Benchmark => &mut self.benchmark_pane,
        }
    }
}
