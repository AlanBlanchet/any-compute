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

impl From<bool> for CellValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<i64> for CellValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<i32> for CellValue {
    fn from(v: i32) -> Self {
        Self::Int(v as i64)
    }
}
impl From<f64> for CellValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<f32> for CellValue {
    fn from(v: f32) -> Self {
        Self::Float(v as f64)
    }
}
impl From<String> for CellValue {
    fn from(v: String) -> Self {
        Self::Text(v)
    }
}
impl From<&str> for CellValue {
    fn from(v: &str) -> Self {
        Self::Text(v.to_string())
    }
}

impl CellValue {
    /// Try to extract as f64 (Int → f64, Float → f64, Bool → 0.0/1.0).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Float(v) => Some(*v),
            Self::Int(v) => Some(*v as f64),
            Self::Bool(v) => Some(if *v { 1.0 } else { 0.0 }),
            _ => None,
        }
    }

    /// Try to extract as i64 (Int → i64, Bool → 0/1).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(v) => Some(*v),
            Self::Bool(v) => Some(if *v { 1 } else { 0 }),
            _ => None,
        }
    }

    /// Try to extract as string reference.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Text(v) => Some(v),
            _ => None,
        }
    }
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
                ColumnMeta {
                    name: "id".into(),
                    kind: ColumnKind::Int,
                },
                ColumnMeta {
                    name: "val".into(),
                    kind: ColumnKind::Float,
                },
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
        let src = VecSource {
            columns: vec![],
            rows: vec![],
        };
        assert_eq!(src.row_count(), 0);
        assert_eq!(src.fetch(0..10).len(), 0);
    }

    // ── CellValue conversions ───────────────────────────────────────────

    #[test]
    fn cell_from_primitives() {
        assert_eq!(CellValue::from(true), CellValue::Bool(true));
        assert_eq!(CellValue::from(42i64), CellValue::Int(42));
        assert_eq!(CellValue::from(7i32), CellValue::Int(7));
        assert_eq!(CellValue::from(3.14f64), CellValue::Float(3.14));
        assert_eq!(CellValue::from(1.5f32), CellValue::Float(1.5));
        assert_eq!(CellValue::from("hello"), CellValue::Text("hello".into()));
        assert_eq!(
            CellValue::from(String::from("world")),
            CellValue::Text("world".into())
        );
    }

    #[test]
    fn cell_accessors() {
        assert_eq!(CellValue::Float(3.14).as_f64(), Some(3.14));
        assert_eq!(CellValue::Int(42).as_f64(), Some(42.0));
        assert_eq!(CellValue::Bool(true).as_f64(), Some(1.0));
        assert_eq!(CellValue::Text("x".into()).as_f64(), None);
        assert_eq!(CellValue::Int(7).as_i64(), Some(7));
        assert_eq!(CellValue::Bool(false).as_i64(), Some(0));
        assert_eq!(CellValue::Text("hi".into()).as_str(), Some("hi"));
        assert_eq!(CellValue::Int(1).as_str(), None);
    }
}
