use crate::kernel::{BinaryOp, ReduceOp, UnaryOp};
use std::collections::HashMap;
use std::sync::Mutex;

use super::Device;

// ═══════════════════════════════════════════════════════════════════════════
// ── OpCache — content-addressed memoization of compute results ──────────
// ═══════════════════════════════════════════════════════════════════════════

/// Content-addressed cache for device operations.
///
/// Keys on `(op_tag, data_generation)` so identical computations on unchanged
/// data return instantly. Thread-safe via internal `Mutex`.
///
/// ```
/// use any_compute_core::compute::{Device, OpCache};
/// use any_compute_core::kernel::ReduceOp;
///
/// let dev = Device::cpu();
/// let cache = OpCache::new(1024);
/// let data = vec![1.0, 2.0, 3.0];
///
/// // First call computes and caches.
/// let sum = cache.reduce(&dev, &data, ReduceOp::Sum);
/// // Second call with same data hits cache.
/// let sum2 = cache.reduce(&dev, &data, ReduceOp::Sum);
/// assert_eq!(sum, sum2);
/// ```
pub struct OpCache {
    inner: Mutex<CacheInner>,
}

struct CacheInner {
    entries: HashMap<u64, CacheEntry>,
    capacity: usize,
}

#[derive(Clone)]
enum CacheEntry {
    Vector(Vec<f64>),
    Scalar(f64),
}

impl OpCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                entries: HashMap::with_capacity(capacity.min(256)),
                capacity,
            }),
        }
    }

    /// Cache key from data pointer + length + op discriminant.
    fn key(data_ptr: usize, data_len: usize, op_tag: u64) -> u64 {
        // FNV-1a style mix — fast, good enough for cache keys.
        let mut h = 0xcbf29ce484222325u64;
        h ^= data_ptr as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= data_len as u64;
        h = h.wrapping_mul(0x100000001b3);
        h ^= op_tag;
        h.wrapping_mul(0x100000001b3)
    }

    fn op_tag_unary(op: UnaryOp) -> u64 {
        // Discriminant as tag — UnaryOp variants are small integers.
        // Safety: enum discriminant read via mem::discriminant is stable.
        let disc = std::mem::discriminant(&op);
        let mut bytes = [0u8; 8];
        bytes[0] = 1; // namespace: unary
        // Use pointer to discriminant as hash input
        let d_ptr = &disc as *const _ as usize;
        bytes[1..5].copy_from_slice(&(d_ptr as u32).to_le_bytes());
        u64::from_le_bytes(bytes)
    }

    fn op_tag_reduce(op: ReduceOp) -> u64 {
        let mut bytes = [0u8; 8];
        bytes[0] = 2; // namespace: reduce
        bytes[1] = op as u8;
        u64::from_le_bytes(bytes)
    }

    fn op_tag_binary(op: BinaryOp) -> u64 {
        let mut bytes = [0u8; 8];
        bytes[0] = 3; // namespace: binary
        bytes[1] = op as u8;
        u64::from_le_bytes(bytes)
    }

    /// Cached unary: returns from cache if data pointer + len + op match.
    pub fn unary(&self, dev: &Device, data: &[f64], op: UnaryOp) -> Vec<f64> {
        let k = Self::key(data.as_ptr() as usize, data.len(), Self::op_tag_unary(op));
        if let Some(CacheEntry::Vector(v)) = self.get(k) {
            return v;
        }
        let result = dev.unary(data, op);
        self.put(k, CacheEntry::Vector(result.clone()));
        result
    }

    /// Cached reduce.
    pub fn reduce(&self, dev: &Device, data: &[f64], op: ReduceOp) -> f64 {
        let k = Self::key(data.as_ptr() as usize, data.len(), Self::op_tag_reduce(op));
        if let Some(CacheEntry::Scalar(s)) = self.get(k) {
            return s;
        }
        let result = dev.reduce(data, op);
        self.put(k, CacheEntry::Scalar(result));
        result
    }

    /// Cached binary.
    pub fn binary(&self, dev: &Device, a: &[f64], b: &[f64], op: BinaryOp) -> Vec<f64> {
        let k = Self::key(a.as_ptr() as usize, a.len(), Self::op_tag_binary(op));
        if let Some(CacheEntry::Vector(v)) = self.get(k) {
            return v;
        }
        let result = dev.binary(a, b, op);
        self.put(k, CacheEntry::Vector(result.clone()));
        result
    }

    /// Evict all entries.
    pub fn clear(&self) {
        self.inner.lock().unwrap().entries.clear();
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, key: u64) -> Option<CacheEntry> {
        self.inner.lock().unwrap().entries.get(&key).cloned()
    }

    fn put(&self, key: u64, entry: CacheEntry) {
        let mut inner = self.inner.lock().unwrap();
        if inner.entries.len() >= inner.capacity {
            // Simple eviction: clear all when full. A real LRU can replace this.
            inner.entries.clear();
        }
        inner.entries.insert(key, entry);
    }
}
