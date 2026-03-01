//! Virtualized data access — the key to handling millions of rows without blowing up memory.
//!
//! Consumers implement [`DataSource`] to feed any-compute with data.
//! The engine only ever requests the *visible window* of rows,
//! so the backing store can be lazy, streamed, or memory-mapped.

use std::ops::Range;

/// A single cell value — kept small and Copy-friendly.
#[derive(Debug, Clone, PartialEq)]
pub enum CellValue {
    Empty,
    Bool(bool),
    Int(i64),
    Float(f64),
    Text(String),
}

/// Metadata for one column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnMeta {
    pub name: String,
    pub kind: ColumnKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnKind {
    Bool,
    Int,
    Float,
    Text,
}

/// Trait that any data backend implements.
///
/// The engine only calls [`fetch`] for the rows it actually needs to paint,
/// enabling virtualized rendering of arbitrarily large datasets.
pub trait DataSource: Send + Sync {
    /// Total number of rows (may be approximate for streaming sources).
    fn row_count(&self) -> usize;

    /// Column definitions.
    fn columns(&self) -> &[ColumnMeta];

    /// Fetch a window of rows. `rows` is a half-open range.
    /// Returns one `Vec<CellValue>` per row, each with `columns().len()` entries.
    fn fetch(&self, rows: Range<usize>) -> Vec<Vec<CellValue>>;
}

/// In-memory data source backed by a flat `Vec`.
/// Good for small-to-medium datasets or testing.
#[derive(Debug, Clone, PartialEq)]
pub struct VecSource {
    pub columns: Vec<ColumnMeta>,
    pub rows: Vec<Vec<CellValue>>,
}

impl DataSource for VecSource {
    fn row_count(&self) -> usize {
        self.rows.len()
    }

    fn columns(&self) -> &[ColumnMeta] {
        &self.columns
    }

    fn fetch(&self, range: Range<usize>) -> Vec<Vec<CellValue>> {
        let end = range.end.min(self.rows.len());
        let start = range.start.min(end);
        self.rows[start..end].to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_source() -> VecSource {
        VecSource {
            columns: vec![
                ColumnMeta { name: "id".into(), kind: ColumnKind::Int },
                ColumnMeta { name: "val".into(), kind: ColumnKind::Float },
            ],
            rows: (0..100)
                .map(|i| vec![CellValue::Int(i), CellValue::Float(i as f64 * 0.5)])
                .collect(),
        }
    }

    #[test]
    fn row_count() {
        assert_eq!(sample_source().row_count(), 100);
    }

    #[test]
    fn columns_meta() {
        let src = sample_source();
        assert_eq!(src.columns().len(), 2);
        assert_eq!(src.columns()[0].kind, ColumnKind::Int);
    }

    #[test]
    fn fetch_window() {
        let src = sample_source();
        let rows = src.fetch(10..15);
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0][0], CellValue::Int(10));
    }

    #[test]
    fn fetch_clamps_to_bounds() {
        let src = sample_source();
        assert_eq!(src.fetch(95..200).len(), 5);
        assert_eq!(src.fetch(200..300).len(), 0);
    }

    #[test]
    fn empty_source() {
        let src = VecSource { columns: vec![], rows: vec![] };
        assert_eq!(src.row_count(), 0);
        assert_eq!(src.fetch(0..10).len(), 0);
    }
}
