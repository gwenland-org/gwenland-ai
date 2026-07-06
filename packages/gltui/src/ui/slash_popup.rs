use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Style},
    widgets::{Block, Borders, List, ListItem, ListState, Paragraph, Clear},
    Frame,
};
use crossterm::event::{KeyCode, KeyEvent};
use tui_textarea::TextArea;
use crate::app::PaneId;
use crate::ui::theme;
use std::path::PathBuf;

pub enum SlashAction {
    None,
    Handled,
    Close,
    SwitchPane(PaneId),
    ClearHistory,
}

enum PopupState {
    Commands,
    Settings,
}

pub struct SlashPopup<'a> {
    pub active: bool,
    state: PopupState,
    commands: Vec<(&'static str, &'static str, &'static str)>,
    filtered_commands: Vec<(&'static str, &'static str, &'static str)>,
    list_state: ListState,
    
    // Settings form fields
    model_dir: TextArea<'a>,
    default_model: TextArea<'a>,
    temp: TextArea<'a>,
    top_k: TextArea<'a>,
    top_p: TextArea<'a>,
    settings_index: usize,
}

impl<'a> SlashPopup<'a> {
    pub fn new() -> Self {
        let commands = vec![
            ("/chat", "switch to chat", "ctrl+c"),
            ("/train", "switch to train", "ctrl+t"),
            ("/fetch", "switch to fetch", "ctrl+f"),
            ("/engines", "switch to engines", "ctrl+e"),
            ("/bench", "switch to benchmark", "ctrl+b"),
            ("/model", "pick active model", ""),
            ("/engine", "switch active engine", ""),
            ("/clear", "clear chat history", ""),
            ("/settings", "open inline settings", ""),
        ];
        
        let mut sl = Self {
            active: false,
            state: PopupState::Commands,
            filtered_commands: commands.clone(),
            commands,
            list_state: ListState::default(),
            model_dir: TextArea::default(),
            default_model: TextArea::default(),
            temp: TextArea::default(),
            top_k: TextArea::default(),
            top_p: TextArea::default(),
            settings_index: 0,
        };
        for ta in [&mut sl.model_dir, &mut sl.default_model, &mut sl.temp, &mut sl.top_k, &mut sl.top_p] {
            ta.set_style(Style::default().fg(theme::TEXT).bg(theme::BG));
            ta.set_cursor_line_style(Style::default().bg(theme::BG));
        }
        sl.list_state.select(Some(0));
        sl
    }
    
    pub fn open(&mut self) {
        self.active = true;
        self.state = PopupState::Commands;
        self.update_filter("");
    }
    
    pub fn close(&mut self) {
        self.active = false;
    }
    
    pub fn update_filter(&mut self, query: &str) {
        let q = query.strip_prefix('/').unwrap_or(query).to_lowercase();
        self.filtered_commands = self.commands.iter()
            .filter(|(cmd, desc, _)| cmd.to_lowercase().contains(&q) || desc.to_lowercase().contains(&q))
            .copied()
            .collect();
        self.list_state.select(if self.filtered_commands.is_empty() { None } else { Some(0) });
    }
    
    pub fn draw(&mut self, f: &mut Frame, area: Rect) {
        if !self.active { return; }
        
        f.render_widget(Clear, area);
        let block = Block::default()
            .padding(ratatui::widgets::Padding::symmetric(2, 1))
            .style(Style::default().bg(theme::SURFACE));
            
        let inner = block.inner(area);
        f.render_widget(block, area);
        
        match self.state {
            PopupState::Commands => {
                let list_items: Vec<ListItem> = self.filtered_commands.iter().enumerate().map(|(i, (cmd, desc, key))| {
                    let style = if Some(i) == self.list_state.selected() {
                        Style::default().bg(theme::PRIMARY).fg(Color::Black)
                    } else {
                        Style::default().fg(theme::TEXT)
                    };
                    
                    let w = inner.width as usize;
                    let c = format!("{} - {}", cmd, desc);
                    let pad = w.saturating_sub(c.len() + key.len());
                    let content = format!("{}{}{}", c, " ".repeat(pad), key);
                    ListItem::new(content).style(style)
                }).collect();
                
                let list = List::new(list_items);
                f.render_stateful_widget(list, inner, &mut self.list_state);
            }
            PopupState::Settings => {
                let chunks = Layout::default()
                    .direction(Direction::Vertical)
                    .constraints([
                        Constraint::Length(1), // Title
                        Constraint::Length(1), // padding
                        Constraint::Length(1), // Model Dir
                        Constraint::Length(1), // padding
                        Constraint::Length(1), // Default Model
                        Constraint::Length(1), // padding
                        Constraint::Length(1), // Temp
                        Constraint::Length(1), // padding
                        Constraint::Length(1), // Top-K
                        Constraint::Length(1), // padding
                        Constraint::Length(1), // Top-P
                        Constraint::Min(0),    // Save hint
                    ])
                    .split(inner);
                    
                f.render_widget(Paragraph::new("Settings (Esc to cancel, Enter to save)").style(Style::default().fg(theme::PRIMARY).add_modifier(ratatui::style::Modifier::BOLD)), chunks[0]);
                
                let fields = [
                    ("Model Directory", &self.model_dir, 2),
                    ("Default Model", &self.default_model, 4),
                    ("Temperature", &self.temp, 6),
                    ("Top-K", &self.top_k, 8),
                    ("Top-P", &self.top_p, 10),
                ];
                
                for (i, (title, textarea, row_idx)) in fields.iter().enumerate() {
                    let is_active = i == self.settings_index;
                    let label_style = if is_active { Style::default().fg(theme::PRIMARY).add_modifier(ratatui::style::Modifier::BOLD) } else { Style::default().fg(theme::MUTED) };
                    
                    let field_layout = Layout::default()
                        .direction(Direction::Horizontal)
                        .constraints([Constraint::Length(24), Constraint::Min(0)])
                        .split(chunks[*row_idx]);
                        
                    f.render_widget(Paragraph::new(title.to_string()).style(label_style), field_layout[0]);
                    
                    let mut ta = (*textarea).clone();
                    let bg = if is_active { theme::BG } else { theme::SURFACE };
                    ta.set_block(Block::default().style(Style::default().bg(bg)));
                    ta.set_style(Style::default().fg(theme::TEXT).bg(bg));
                    ta.set_cursor_line_style(Style::default().bg(bg));
                    f.render_widget(&ta, field_layout[1]);
                }
            }
        }
    }
    
    pub fn handle_key(&mut self, key: KeyEvent, query: &mut String) -> SlashAction {
        match self.state {
            PopupState::Commands => self.handle_commands_key(key, query),
            PopupState::Settings => self.handle_settings_key(key),
        }
    }
    
    fn handle_commands_key(&mut self, key: KeyEvent, _query: &mut String) -> SlashAction {
        match key.code {
            KeyCode::Esc => SlashAction::Close,
            KeyCode::Down => {
                if !self.filtered_commands.is_empty() {
                    let i = match self.list_state.selected() {
                        Some(i) => (i + 1) % self.filtered_commands.len(),
                        None => 0,
                    };
                    self.list_state.select(Some(i));
                }
                SlashAction::Handled
            }
            KeyCode::Up => {
                if !self.filtered_commands.is_empty() {
                    let i = match self.list_state.selected() {
                        Some(i) => if i == 0 { self.filtered_commands.len() - 1 } else { i - 1 },
                        None => 0,
                    };
                    self.list_state.select(Some(i));
                }
                SlashAction::Handled
            }
            KeyCode::Enter => {
                if let Some(i) = self.list_state.selected() {
                    if let Some((cmd, _, _)) = self.filtered_commands.get(i) {
                        return self.execute_command(cmd);
                    }
                }
                SlashAction::Close
            }
            _ => SlashAction::None, // Not handled, ChatPane feeds to textarea
        }
    }
    
    fn execute_command(&mut self, cmd: &str) -> SlashAction {
        match cmd {
            "/chat" => SlashAction::SwitchPane(PaneId::Chat),
            "/train" => SlashAction::SwitchPane(PaneId::Train),
            "/fetch" => SlashAction::SwitchPane(PaneId::Fetch),
            "/engines" => SlashAction::SwitchPane(PaneId::Engines),
            "/bench" => SlashAction::SwitchPane(PaneId::Benchmark),
            "/clear" => SlashAction::ClearHistory,
            "/settings" => {
                self.open_settings();
                SlashAction::None
            }
            _ => SlashAction::Close,
        }
    }
    
    fn open_settings(&mut self) {
        self.state = PopupState::Settings;
        self.settings_index = 0;
        
        let config_path = dirs::home_dir().unwrap_or(PathBuf::from(".")).join(".gwenland").join("config.toml");
        
        if let Ok(content) = std::fs::read_to_string(&config_path) {
            if let Ok(val) = content.parse::<toml::Value>() {
                if let Some(mdir) = val.get("model_dir").and_then(|v| v.as_str()) {
                    self.model_dir = TextArea::from(vec![mdir.to_string()].into_iter());
                }
                if let Some(dmodel) = val.get("default_model").and_then(|v| v.as_str()) {
                    self.default_model = TextArea::from(vec![dmodel.to_string()].into_iter());
                }
                if let Some(params) = val.get("sampler") {
                    if let Some(t) = params.get("temperature").and_then(|v| v.as_float()) {
                        self.temp = TextArea::from(vec![t.to_string()].into_iter());
                    }
                    if let Some(tk) = params.get("top_k").and_then(|v| v.as_integer()) {
                        self.top_k = TextArea::from(vec![tk.to_string()].into_iter());
                    }
                    if let Some(tp) = params.get("top_p").and_then(|v| v.as_float()) {
                        self.top_p = TextArea::from(vec![tp.to_string()].into_iter());
                    }
                }
            }
        }
        
        for ta in [&mut self.model_dir, &mut self.default_model, &mut self.temp, &mut self.top_k, &mut self.top_p] {
            ta.set_style(Style::default().fg(theme::TEXT).bg(theme::BG));
            ta.set_cursor_line_style(Style::default().bg(theme::BG));
        }
    }
    
    fn save_settings(&self) {
        let config_path = dirs::home_dir().unwrap_or(PathBuf::from(".")).join(".gwenland").join("config.toml");
        std::fs::create_dir_all(config_path.parent().unwrap()).unwrap_or(());
        
        let mut toml = toml::map::Map::new();
        toml.insert("model_dir".to_string(), toml::Value::String(self.model_dir.lines().join("")));
        toml.insert("default_model".to_string(), toml::Value::String(self.default_model.lines().join("")));
        
        let mut sampler = toml::map::Map::new();
        if let Ok(t) = self.temp.lines().join("").parse::<f64>() {
            sampler.insert("temperature".to_string(), toml::Value::Float(t));
        }
        if let Ok(tk) = self.top_k.lines().join("").parse::<i64>() {
            sampler.insert("top_k".to_string(), toml::Value::Integer(tk));
        }
        if let Ok(tp) = self.top_p.lines().join("").parse::<f64>() {
            sampler.insert("top_p".to_string(), toml::Value::Float(tp));
        }
        toml.insert("sampler".to_string(), toml::Value::Table(sampler));
        
        if let Ok(s) = toml::to_string_pretty(&toml::Value::Table(toml)) {
            let _ = std::fs::write(&config_path, s);
        }
    }
    
    fn handle_settings_key(&mut self, key: KeyEvent) -> SlashAction {
        match key.code {
            KeyCode::Esc => SlashAction::Close,
            KeyCode::Enter => {
                self.save_settings();
                SlashAction::Close
            }
            KeyCode::Tab | KeyCode::Down => {
                self.settings_index = (self.settings_index + 1) % 5;
                SlashAction::Handled
            }
            KeyCode::BackTab | KeyCode::Up => {
                self.settings_index = if self.settings_index == 0 { 4 } else { self.settings_index - 1 };
                SlashAction::Handled
            }
            _ => {
                match self.settings_index {
                    0 => { self.model_dir.input(key); }
                    1 => { self.default_model.input(key); }
                    2 => { self.temp.input(key); }
                    3 => { self.top_k.input(key); }
                    4 => { self.top_p.input(key); }
                    _ => {}
                }
                SlashAction::Handled
            }
        }
    }
}
