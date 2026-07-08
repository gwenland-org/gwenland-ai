pub mod app;
pub mod core_bridge;
pub mod events;
pub mod panes;
pub mod terminal;
pub mod ui;

use app::App;
use clap::Parser;
use std::error::Error;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long)]
    chat: bool,
    #[arg(long)]
    train: bool,
    #[arg(long)]
    fetch: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();
    
    let initial_pane = if args.chat {
        app::PaneId::Chat
    } else if args.train {
        app::PaneId::Train
    } else if args.fetch {
        app::PaneId::Fetch
    } else {
        app::PaneId::Chat
    };

    let mut terminal = terminal::setup()?;
    let mut app = App::new(initial_pane);
    
    let res = app.run(&mut terminal).await;

    terminal::teardown(&mut terminal)?;

    if let Err(err) = res {
        println!("{:?}", err);
    }

    Ok(())
}

#[cfg(test)]
mod render_snapshot {
    //! Headless render snapshots: draw each redesigned screen into a ratatui
    //! TestBackend and print the resulting pane. Run with:
    //!   cargo test -p gltui --bin gltui -- --nocapture render
    //! These assert nothing structural beyond "the draw pipeline runs and
    //! produces a non-empty pane"; the value is the printed capture.
    use crate::app::{App, PaneId};
    use crate::ui::layout::RootLayout;
    use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
    use ratatui::{backend::TestBackend, buffer::Buffer, Terminal};

    fn key(c: char) -> KeyEvent {
        KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE)
    }

    fn render(app: &mut App, w: u16, h: u16) -> Buffer {
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        term.draw(|f| RootLayout::draw(f, app)).unwrap();
        term.backend().buffer().clone()
    }

    fn dump(title: &str, buf: &Buffer) {
        let area = buf.area();
        println!("\n===== {title} ({}x{}) =====", area.width, area.height);
        for y in 0..area.height {
            let mut line = String::new();
            for x in 0..area.width {
                let s = buf.cell((x, y)).unwrap().symbol();
                line.push_str(if s.is_empty() { " " } else { s });
            }
            println!("|{}|", line.trim_end());
        }
    }

    /// True if any cell in the buffer carries this exact foreground color.
    fn has_fg(buf: &Buffer, color: ratatui::style::Color) -> bool {
        let area = buf.area();
        (0..area.height).any(|y| {
            (0..area.width).any(|x| buf.cell((x, y)).unwrap().style().fg == Some(color))
        })
    }

    fn buf_text(buf: &Buffer) -> String {
        let area = buf.area();
        let mut s = String::new();
        for y in 0..area.height {
            for x in 0..area.width {
                s.push_str(buf.cell((x, y)).unwrap().symbol());
            }
            s.push('\n');
        }
        s
    }

    #[test]
    fn welcome_screen() {
        let mut app = App::new(PaneId::Chat);
        let buf = render(&mut app, 80, 24);
        dump("Welcome (Chat)", &buf);
        let text = buf_text(&buf);
        // ASCII-art logo is gone; plain title present.
        assert!(text.contains("GwenLand"));
        assert!(!text.contains("____"), "ASCII art must be removed");
        assert!(text.contains("Local AI. Your machine."));
        // Rounded box corner rendered.
        assert!(text.contains('╭') || text.contains('╮'), "rounded card border");
    }

    #[test]
    fn slash_palette() {
        let mut app = App::new(PaneId::Chat);
        app.active_pane_mut().handle_key(key('/'));
        let buf = render(&mut app, 80, 24);
        dump("Slash palette (typed '/')", &buf);
        let text = buf_text(&buf);
        assert!(text.contains("/chat"));
        assert!(text.contains("/settings"));
        // Selection bg color (orange accent) appears on the highlighted row.
        assert!(has_fg(&buf, crate::ui::theme::SELECTION_FG));
    }

    #[test]
    fn slash_palette_filtered() {
        let mut app = App::new(PaneId::Chat);
        for c in "/mo".chars() {
            app.active_pane_mut().handle_key(key(c));
        }
        let buf = render(&mut app, 80, 24);
        dump("Slash palette (typed '/mo')", &buf);
        let text = buf_text(&buf);
        assert!(text.contains("/model"), "filter keeps matching command");
        assert!(!text.contains("/chat"), "filter drops non-matching command");
    }

    #[test]
    fn status_bar_active_tab() {
        let mut app = App::new(PaneId::Train);
        let buf = render(&mut app, 80, 24);
        dump("Train pane (status bar active=train)", &buf);
        let text = buf_text(&buf);
        assert!(text.contains("chat") && text.contains("train"));
        assert!(text.contains("GwenLand") && text.contains("v1.0"));
        // Accent color used somewhere (active tab).
        assert!(has_fg(&buf, crate::ui::theme::ACCENT));
    }
}
