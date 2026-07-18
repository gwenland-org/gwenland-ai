use std::time::Duration;
use tokio::sync::mpsc;
use crossterm::event::{self, Event as CrosstermEvent, KeyEvent, MouseEvent};

pub enum Event {
    Tick,
    Key(KeyEvent),
    Mouse(MouseEvent),
    Resize(u16, u16),
}

pub struct EventHandler {
    receiver: mpsc::UnboundedReceiver<Event>,
}

impl EventHandler {
    pub fn new(tick_rate: u64) -> Self {
        let (sender, receiver) = mpsc::unbounded_channel();
        let tick_rate = Duration::from_millis(tick_rate);
        
        let sender_clone = sender.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tick_rate).await;
                if sender_clone.send(Event::Tick).is_err() {
                    break;
                }
            }
        });

        tokio::task::spawn_blocking(move || {
            loop {
                if event::poll(Duration::from_millis(100)).unwrap_or(false) {
                    if let Ok(evt) = event::read() {
                        let res = match evt {
                            CrosstermEvent::Key(key) => {
                                if key.kind == crossterm::event::KeyEventKind::Press {
                                    sender.send(Event::Key(key))
                                } else {
                                    Ok(())
                                }
                            },
                            CrosstermEvent::Mouse(mouse) => sender.send(Event::Mouse(mouse)),
                            CrosstermEvent::Resize(w, h) => sender.send(Event::Resize(w, h)),
                            _ => Ok(()),
                        };
                        if res.is_err() {
                            break;
                        }
                    }
                }
            }
        });

        Self { receiver }
    }

    pub async fn next(&mut self) -> Result<Event, std::io::Error> {
        self.receiver.recv().await.ok_or(std::io::Error::new(
            std::io::ErrorKind::Other,
            "Event channel closed",
        ))
    }
}
