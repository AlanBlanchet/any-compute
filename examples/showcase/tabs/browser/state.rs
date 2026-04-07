use super::*;
pub(super) const DEVTABS: &[&str] = &["Elements", "Console", "Network", "Source"];

/// Tag for the page preview viewport node (used by main.rs for overlay + events).
pub const VIEWPORT_TAG: &str = "browser-viewport";

// ═══════════════════════════════════════════════════════════════════════════
// Browser state
// ═══════════════════════════════════════════════════════════════════════════

/// Tag for the DevTools right panel.
pub const DEVTOOLS_TAG: &str = "browser-devtools";

/// Color constants for the browser chrome.
pub(super) mod chrome {
    use any_compute_core::render::Color;
    use any_compute_dom::theme;
    pub const TAB_BG: Color = Color::rgba(35, 39, 52, 255);
    pub const TAB_ACTIVE: Color = theme::BG;
    pub const NAV_BG: Color = Color::rgba(42, 46, 60, 255);
    pub const LOCK_GREEN: Color = Color::rgba(80, 200, 80, 255);
}

pub struct BrowserState {
    pub page: Option<Page>,
    pub cursor: Point,
    pub needs_measure: bool,
    pub html: String,
    /// Editable URL bar.
    pub input: TextInput,
    /// Console log entries (level, message).
    pub console_logs: Vec<(String, String)>,
    /// Network request log (method, url, status, size).
    pub network_log: Vec<(String, String, u16, usize)>,
    /// Scroll position for the DevTools panel.
    pub devtools_scroll: f64,
    /// Whether DevTools panel is open.
    pub devtools_open: bool,
    /// Navigation can_go_back / can_go_forward.
    pub history: Vec<String>,
    pub history_idx: i32,
    /// Active browser tab (we show one tab but support the concept).
    pub tab_title: String,
    /// Whether the URL is HTTPS.
    pub is_secure: bool,
}

impl Default for BrowserState {
    fn default() -> Self {
        Self {
            page: None,
            cursor: Point::ZERO,
            needs_measure: true,
            html: String::new(),
            input: TextInput::new("dashboard"),
            console_logs: vec![
                ("info".into(), "Page loaded".into()),
                ("info".into(), "any-compute engine v0.1.0".into()),
            ],
            network_log: Vec::new(),
            devtools_scroll: 0.0,
            devtools_open: true,
            history: Vec::new(),
            history_idx: -1,
            tab_title: "Dashboard".into(),
            is_secure: false,
        }
    }
}

impl BrowserState {
    /// Reload the page from the current HTML source.
    pub fn reload(&mut self, css: &str) {
        log::info!("browser: reloading page ({} chars HTML)", self.html.len());
        let html = self.html.clone();
        let css = css.to_string();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            Page::load_with(&html, &css, &[])
        })) {
            Ok(page) => {
                self.page = Some(page);
                self.needs_measure = true;
            }
            Err(e) => {
                log::error!("Page load panicked: {:?}", e.downcast_ref::<&str>());
                self.page = Some(Page::load(ERROR_PAGE_HTML));
                self.needs_measure = true;
            }
        }
    }

    /// Focus the URL bar.
    pub fn focus(&mut self) {
        self.input.focus();
    }

    /// Toggle DevTools panel visibility.
    pub fn toggle_devtools(&mut self) {
        self.devtools_open = !self.devtools_open;
    }

    /// Navigate back in history.
    pub fn go_back(&mut self, css: &str) {
        if self.history_idx > 0 {
            self.history_idx -= 1;
            let url = self.history[self.history_idx as usize].clone();
            self.input.set_text(&url);
            self.reload(css);
        }
    }

    /// Navigate forward in history.
    pub fn go_forward(&mut self, css: &str) {
        if (self.history_idx as usize) + 1 < self.history.len() {
            self.history_idx += 1;
            let url = self.history[self.history_idx as usize].clone();
            self.input.set_text(&url);
            self.reload(css);
        }
    }

    /// Navigate to the current URL. Returns the URL for external fetch if needed.
    pub fn navigate(&mut self, css: &str) -> Option<String> {
        let url = self.input.text.trim().to_string();
        log::info!("browser: navigate → {url}");

        // Push to history
        self.history
            .truncate((self.history_idx + 1).max(0) as usize);
        self.history.push(url.clone());
        self.history_idx = self.history.len() as i32 - 1;

        // Update title from URL
        self.tab_title = url
            .split('/')
            .filter(|s| !s.is_empty() && !s.starts_with("http"))
            .last()
            .unwrap_or("Page")
            .to_string();
        self.is_secure = url.starts_with("https://");

        // file:// protocol — load local HTML file
        if let Some(path) = url.strip_prefix("file://") {
            match std::fs::read_to_string(path) {
                Ok(content) => {
                    let size = content.len();
                    self.network_log
                        .push(("FILE".into(), url.clone(), 200, size));
                    self.console_logs
                        .push(("info".into(), format!("Loaded file: {path}")));
                    self.html = content;
                    self.tab_title = path.rsplit('/').next().unwrap_or("file").into();
                    self.reload(css);
                }
                Err(e) => {
                    log::error!("file:// load failed: {e}");
                    self.console_logs
                        .push(("error".into(), format!("Failed to load: {e}")));
                    self.html = format!(
                        "<h1>File Error</h1><p>{}</p>",
                        e.to_string()
                            .replace('&', "&amp;")
                            .replace('<', "&lt;")
                            .replace('>', "&gt;")
                    );
                    self.reload(css);
                }
            }
            return None;
        }
        if url.starts_with("http://") || url.starts_with("https://") {
            return Some(url);
        }
        // Domain-like input (e.g. "google.com") → prepend https://
        if url.contains('.') && !url.contains(' ') {
            let full = format!("https://{url}");
            self.input.set_text(&full);
            self.is_secure = true;
            return Some(full);
        }
        self.reload(css);
        None
    }

    /// Load fetched HTML from a URL.
    pub fn load_fetched(&mut self, html: String, css: &str) {
        let url = self.input.text.clone();
        let size = html.len();
        self.network_log
            .push(("GET".into(), url.clone(), 200, size));
        self.console_logs
            .push(("info".into(), format!("Navigated to {url}")));
        self.html = html;
        self.reload(css);
    }

    /// Node count in the loaded page.
    pub fn node_count(&self) -> usize {
        self.page.as_ref().map_or(0, |p| p.tree().arena.len())
    }

    pub(super) fn can_go_back(&self) -> bool {
        self.history_idx > 0
    }

    pub(super) fn can_go_forward(&self) -> bool {
        self.history_idx >= 0 && (self.history_idx as usize) + 1 < self.history.len()
    }
}
