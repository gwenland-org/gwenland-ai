//! A tiny fixed-width table renderer for terminal output — no dependencies.
//!
//! Computes per-column widths and left/right-aligns cells. Used by
//! [`super::text`] to lay out throughput stats and comparison deltas.

/// A simple text table: a header row plus body rows, all same column count.
#[derive(Debug, Default)]
pub struct Table {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
    /// Right-align flag per column (numbers read better right-aligned).
    right: Vec<bool>,
}

impl Table {
    /// Start a table with the given column headers. All columns default to
    /// left-aligned; call [`Table::right_align`] to change one.
    pub fn new(headers: &[&str]) -> Table {
        Table {
            headers: headers.iter().map(|h| h.to_string()).collect(),
            rows: Vec::new(),
            right: vec![false; headers.len()],
        }
    }

    /// Mark column `col` right-aligned.
    pub fn right_align(mut self, col: usize) -> Self {
        if col < self.right.len() {
            self.right[col] = true;
        }
        self
    }

    /// Append a row. Extra cells are ignored; missing cells render blank.
    pub fn row(&mut self, cells: &[String]) {
        self.rows.push(cells.to_vec());
    }

    /// Render the table to a string with a header underline.
    pub fn render(&self) -> String {
        let cols = self.headers.len();
        let mut width = vec![0usize; cols];
        for (c, h) in self.headers.iter().enumerate() {
            width[c] = h.chars().count();
        }
        for row in &self.rows {
            for (w, cell) in width.iter_mut().zip(row.iter()) {
                *w = (*w).max(cell.chars().count());
            }
        }

        let mut out = String::new();
        self.emit_row(&mut out, &self.headers, &width);
        // Underline.
        let mut sep = Vec::with_capacity(cols);
        for w in &width {
            sep.push("-".repeat(*w));
        }
        self.emit_row(&mut out, &sep, &width);
        for row in &self.rows {
            let cells: Vec<String> = (0..cols).map(|c| row.get(c).cloned().unwrap_or_default()).collect();
            self.emit_row(&mut out, &cells, &width);
        }
        out
    }

    fn emit_row(&self, out: &mut String, cells: &[String], width: &[usize]) {
        for (c, cell) in cells.iter().enumerate() {
            if c > 0 {
                out.push_str("  ");
            }
            let pad = width[c].saturating_sub(cell.chars().count());
            if self.right.get(c).copied().unwrap_or(false) {
                for _ in 0..pad {
                    out.push(' ');
                }
                out.push_str(cell);
            } else {
                out.push_str(cell);
                for _ in 0..pad {
                    out.push(' ');
                }
            }
        }
        out.push('\n');
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_aligned() {
        let mut t = Table::new(&["phase", "tps"]).right_align(1);
        t.row(&["decode".into(), "29.3".into()]);
        t.row(&["prefill".into(), "1439.9".into()]);
        let out = t.render();
        // Header, underline, two rows.
        assert_eq!(out.lines().count(), 4);
        assert!(out.contains("prefill"));
    }
}
