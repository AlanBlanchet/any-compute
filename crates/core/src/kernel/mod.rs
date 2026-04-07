//! Low-level compute kernels — CUDA, ROCm, CPU SIMD, and vendor-specific backends.
//!
//! This module defines the [`Kernel`] trait for dispatch-ready compute operations,
//! plus vendor-specific kernel implementations behind feature flags.

mod cpu;
mod ops;
mod traits;

pub use cpu::{CpuSimdKernel, best_kernel};
pub use ops::*;
pub use traits::*;

#[cfg(feature = "cuda")]
pub use cpu::CudaKernel;
#[cfg(feature = "mkl")]
pub use cpu::MklKernel;
#[cfg(feature = "rocm")]
pub use cpu::RocmKernel;

#[cfg(test)]
mod tests {
    use super::*;

    fn kernel() -> CpuSimdKernel {
        CpuSimdKernel::default()
    }

    #[test]
    fn detect_simd_returns_nonzero_width() {
        let k = kernel();
        assert!(k.vector_width() >= 1);
        assert!(!k.name().is_empty());
    }

    #[test]
    fn map_unary_neg() {
        let k = kernel();
        let data = vec![1.0, -2.0, 3.0];
        let out = k.map_unary_f64(&data, UnaryOp::Neg);
        assert_eq!(out, vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    fn map_unary_relu() {
        let k = kernel();
        let data = vec![-1.0, 0.0, 3.0, -5.0];
        let out = k.map_unary_f64(&data, UnaryOp::Relu);
        assert_eq!(out, vec![0.0, 0.0, 3.0, 0.0]);
    }

    #[test]
    fn map_unary_sigmoid() {
        let k = kernel();
        let out = k.map_unary_f64(&[0.0], UnaryOp::Sigmoid);
        assert!((out[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn map_binary_add() {
        let k = kernel();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let out = k.map_binary_f64(&a, &b, BinaryOp::Add);
        assert_eq!(out, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn reduce_sum() {
        let k = kernel();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        assert!((k.reduce_f64(&data, ReduceOp::Sum) - 10.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_mean() {
        let k = kernel();
        let data = vec![2.0, 4.0, 6.0, 8.0];
        assert!((k.reduce_f64(&data, ReduceOp::Mean) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn reduce_min_max() {
        let k = kernel();
        let data = vec![3.0, 1.0, 4.0, 1.5];
        assert!((k.reduce_f64(&data, ReduceOp::Min) - 1.0).abs() < 1e-10);
        assert!((k.reduce_f64(&data, ReduceOp::Max) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn scan_sum() {
        let k = kernel();
        let out = k.scan_f64(&[1.0, 2.0, 3.0, 4.0], ReduceOp::Sum);
        assert_eq!(out, vec![1.0, 3.0, 6.0, 10.0]);
    }

    #[test]
    fn gemm_identity() {
        let k = kernel();
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let c = k.gemm_f64(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn gemm_small() {
        let k = kernel();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = k.gemm_f64(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn sort_f64() {
        let k = kernel();
        let mut data = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        k.sort_f64(&mut data);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn gather_scatter() {
        let k = kernel();
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let gathered = k.gather_f64(&data, &[4, 2, 0]);
        assert_eq!(gathered, vec![50.0, 30.0, 10.0]);

        let scattered = k.scatter_f64(&[99.0, 88.0], &[1, 3], 5);
        assert_eq!(scattered, vec![0.0, 99.0, 0.0, 88.0, 0.0]);
    }

    #[test]
    fn benchmark_op_runs() {
        let k = kernel();
        let stats = k.benchmark_op(&KernelOp::Reduce {
            len: 10_000,
            op: ReduceOp::Sum,
        });
        assert!(stats.duration_us > 0 || stats.flops >= 0.0);
    }

    #[test]
    fn best_kernel_returns_cpu() {
        let k = best_kernel();
        assert!(k.name().contains("CpuSimd") || k.name().contains("Scalar"));
    }

    #[test]
    fn empty_data() {
        let k = kernel();
        assert_eq!(k.map_unary_f64(&[], UnaryOp::Neg), Vec::<f64>::new());
        assert_eq!(k.map_binary_f64(&[], &[], BinaryOp::Add), Vec::<f64>::new());
        assert_eq!(k.scan_f64(&[], ReduceOp::Sum), Vec::<f64>::new());
    }
}
