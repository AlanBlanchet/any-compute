use serde::{Deserialize, Serialize};

// ── Kernel operation descriptors ──────────────────────────────────────────

/// The set of primitive operations a kernel can execute.
///
/// This is an enum rather than separate trait methods so we can batch
/// heterogeneous ops into a single dispatch queue (important for GPU).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KernelOp {
    /// Element-wise: out[i] = f(a[i])
    MapUnary { len: usize, op: UnaryOp },
    /// Element-wise: out[i] = f(a[i], b[i])
    MapBinary { len: usize, op: BinaryOp },
    /// Reduction: scalar = reduce(data, op)
    Reduce { len: usize, op: ReduceOp },
    /// Prefix scan (inclusive)
    Scan { len: usize, op: ReduceOp },
    /// Sort (unstable)
    Sort { len: usize },
    /// Matrix multiply: C = A × B
    Gemm { m: usize, n: usize, k: usize },
    /// FFT (1D, real → complex)
    Fft { len: usize },
    /// Gather: out[i] = data[indices[i]]
    Gather { data_len: usize, index_len: usize },
    /// Scatter: out[indices[i]] = data[i]
    Scatter { data_len: usize, index_len: usize },
}

/// Unary element-wise operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOp {
    Neg,
    Abs,
    Sqrt,
    Rsqrt,
    Exp,
    Log,
    Sin,
    Cos,
    Tanh,
    Relu,
    Sigmoid,
    Floor,
    Ceil,
    /// Multiply by scalar
    Scale(ordered_f64::F64),
    /// Add scalar
    Offset(ordered_f64::F64),
}

impl UnaryOp {
    /// Apply this operation to a single value.
    #[inline]
    pub fn apply(self, v: f64) -> f64 {
        match self {
            Self::Neg => -v,
            Self::Abs => v.abs(),
            Self::Sqrt => v.sqrt(),
            Self::Rsqrt => 1.0 / v.sqrt(),
            Self::Exp => v.exp(),
            Self::Log => v.ln(),
            Self::Sin => v.sin(),
            Self::Cos => v.cos(),
            Self::Tanh => v.tanh(),
            Self::Relu => v.max(0.0),
            Self::Sigmoid => 1.0 / (1.0 + (-v).exp()),
            Self::Floor => v.floor(),
            Self::Ceil => v.ceil(),
            Self::Scale(s) => v * f64::from(s),
            Self::Offset(o) => v + f64::from(o),
        }
    }
}

/// Binary element-wise operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Min,
    Max,
    Pow,
}

/// Reduction operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReduceOp {
    Sum,
    Product,
    Min,
    Max,
    Mean,
}

// ── Ordered f64 for enum storage ──────────────────────────────────────────

mod ordered_f64 {
    use serde::{Deserialize, Serialize};

    /// A wrapper around `f64` that implements `Eq` and `Hash` by bit pattern.
    /// Used inside enum variants so they can derive Eq.
    #[derive(Debug, Clone, Copy, Serialize, Deserialize)]
    pub struct F64(pub f64);

    impl PartialEq for F64 {
        fn eq(&self, other: &Self) -> bool {
            self.0.to_bits() == other.0.to_bits()
        }
    }
    impl Eq for F64 {}

    impl From<f64> for F64 {
        fn from(v: f64) -> Self {
            Self(v)
        }
    }
    impl From<F64> for f64 {
        fn from(v: F64) -> f64 {
            v.0
        }
    }
}

pub use ordered_f64::F64 as Scalar;
