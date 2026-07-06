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
