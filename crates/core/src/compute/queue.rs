use crate::kernel::{BinaryOp, ReduceOp, UnaryOp};

use super::Device;

/// A recorded operation with its data pointers — ready for batched dispatch.
#[derive(Debug, Clone)]
pub enum QueuedOp {
    Unary {
        data: Vec<f64>,
        op: UnaryOp,
    },
    Binary {
        a: Vec<f64>,
        b: Vec<f64>,
        op: BinaryOp,
    },
    Reduce {
        data: Vec<f64>,
        op: ReduceOp,
    },
    Scan {
        data: Vec<f64>,
        op: ReduceOp,
    },
    Sort {
        data: Vec<f64>,
    },
    Gemm {
        a: Vec<f64>,
        b: Vec<f64>,
        m: usize,
        n: usize,
        k: usize,
    },
}

/// Result of a flushed operation — matches the shape of what was queued.
#[derive(Debug, Clone)]
pub enum OpResult {
    Vector(Vec<f64>),
    Scalar(f64),
}

impl OpResult {
    pub fn into_vec(self) -> Vec<f64> {
        match self {
            Self::Vector(v) => v,
            Self::Scalar(s) => vec![s],
        }
    }
    pub fn as_scalar(&self) -> f64 {
        match self {
            Self::Scalar(s) => *s,
            Self::Vector(v) => v[0],
        }
    }
}

/// Records operations lazily, flushes them as a batch to a [`Device`].
///
/// On CPU this provides cache-locality wins (data stays hot between ops).
/// On GPU this will collapse into fewer roundtrips / command buffer submissions.
///
/// ```
/// use any_compute_core::compute::{Device, QueuedOp};
/// use any_compute_core::kernel::UnaryOp;
///
/// let dev = Device::cpu();
/// let mut q = dev.queue();
/// q.push(QueuedOp::Unary { data: vec![1.0, 4.0, 9.0], op: UnaryOp::Sqrt });
/// q.push(QueuedOp::Unary { data: vec![0.0, 1.0, -1.0], op: UnaryOp::Abs });
/// let results = q.flush();
/// assert_eq!(results.len(), 2);
/// ```
pub struct OpQueue {
    device: Device,
    ops: Vec<QueuedOp>,
}

impl OpQueue {
    fn new(device: Device) -> Self {
        Self {
            device,
            ops: Vec::new(),
        }
    }

    /// Enqueue an operation for batched dispatch.
    pub fn push(&mut self, op: QueuedOp) {
        self.ops.push(op);
    }

    /// Number of pending operations.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Execute all queued ops in order, return results. Clears the queue.
    pub fn flush(&mut self) -> Vec<OpResult> {
        let ops = std::mem::take(&mut self.ops);
        ops.into_iter().map(|op| self.dispatch(op)).collect()
    }

    fn dispatch(&self, op: QueuedOp) -> OpResult {
        let d = &self.device;
        match op {
            QueuedOp::Unary { data, op } => OpResult::Vector(d.unary(&data, op)),
            QueuedOp::Binary { a, b, op } => OpResult::Vector(d.binary(&a, &b, op)),
            QueuedOp::Reduce { data, op } => OpResult::Scalar(d.reduce(&data, op)),
            QueuedOp::Scan { data, op } => OpResult::Vector(d.scan(&data, op)),
            QueuedOp::Sort { mut data } => {
                d.sort(&mut data);
                OpResult::Vector(data)
            }
            QueuedOp::Gemm { a, b, m, n, k } => OpResult::Vector(d.gemm(&a, &b, m, n, k)),
        }
    }
}

impl Device {
    /// Create an operation queue for batched dispatch on this device.
    pub fn queue(&self) -> OpQueue {
        OpQueue::new(self.clone())
    }
}
