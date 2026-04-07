//! Catppuccin Mocha palette — single source of truth.
//!
//! CSS owns *styling*; these give Rust code fast, const-time access for
//! dynamic color logic (bar graphs, active/inactive states, wgpu clear).

use any_compute_core::render::Color;

pub const BG: Color = Color::rgb(30, 30, 46);
pub const SURFACE0: Color = Color::rgb(49, 50, 68);
pub const SURFACE_BRIGHT: Color = Color::rgb(69, 71, 90);
pub const OVERLAY0: Color = Color::rgb(108, 112, 134);
pub const SUBTEXT0: Color = Color::rgb(166, 173, 200);
pub const TEXT: Color = Color::rgb(205, 214, 244);
pub const TEXT_DIM: Color = Color::rgb(147, 153, 178);
pub const GREEN: Color = Color::rgb(166, 227, 161);
pub const BLUE: Color = Color::rgb(137, 180, 250);
pub const RED: Color = Color::rgb(243, 139, 168);
pub const YELLOW: Color = Color::rgb(249, 226, 175);
pub const MAUVE: Color = Color::rgb(203, 166, 247);
pub const PEACH: Color = Color::rgb(250, 179, 135);
pub const TEAL: Color = Color::rgb(148, 226, 213);
pub const PINK: Color = Color::rgb(245, 194, 231);
pub const FLAMINGO: Color = Color::rgb(242, 205, 205);
pub const LAVENDER: Color = Color::rgb(180, 190, 254);
pub const SIDEBAR_BG: Color = Color::rgb(24, 24, 37);
pub const ACCENT: Color = BLUE;
pub const BAR_COLORS: [Color; 4] = [GREEN, BLUE, YELLOW, MAUVE];
