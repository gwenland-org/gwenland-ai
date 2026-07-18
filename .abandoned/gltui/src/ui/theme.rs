use ratatui::style::Color;

// GwenLand TUI palette (OpenCode-inspired). One warm accent: #b56936.
// See megaprompt "Color Palette" — every role below maps to that spec.

pub const BG: Color             = Color::Rgb(18, 18, 20);    // near-black background
pub const SURFACE: Color        = Color::Rgb(26, 26, 30);    // slightly lighter surface
pub const BORDER: Color         = Color::Rgb(45, 45, 52);    // subtle border
pub const BORDER_ACTIVE: Color  = Color::Rgb(181, 105, 54);  // warm border #b56936
pub const TEXT: Color           = Color::Rgb(230, 228, 220); // primary off-white
pub const TEXT_SECONDARY: Color = Color::Rgb(120, 118, 110); // muted secondary
pub const ACCENT: Color         = Color::Rgb(181, 105, 54);  // orange accent #b56936
pub const TEXT_DIM: Color       = Color::Rgb(70, 68, 62);    // very muted
pub const INPUT_BG: Color       = Color::Rgb(32, 32, 36);    // input field bg

// Selection: orange bg with dark fg (see SELECTION_FG).
pub const SELECTION_BG: Color = ACCENT;
pub const SELECTION_FG: Color = BG;

// Aliases kept for existing call sites — repointed at the new roles so the
// whole UI shifts consistently without touching unrelated logic.
pub const PRIMARY: Color = ACCENT;
pub const MUTED: Color   = TEXT_SECONDARY;
pub const CYAN: Color    = ACCENT;
pub const PURPLE: Color  = ACCENT;
