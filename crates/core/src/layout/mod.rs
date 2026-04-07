//! Spatial primitives — N-dimensional, generic, renderer-agnostic.

mod vector;
mod region;
mod constraints;
mod matrix;

pub use vector::*;
pub use region::*;
pub use constraints::*;
pub use matrix::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Lerp;

    #[test]
    fn v2_field_access() {
        let v = V::new(3.0, 4.0);
        assert_eq!(v.x, 3.0);
        assert_eq!(v.y, 4.0);
        assert_eq!(v.w(), 3.0);
        assert_eq!(v.h(), 4.0);
        assert_eq!(v[0], 3.0);
        assert_eq!(v[1], 4.0);
    }

    #[test]
    fn v2_mutable_field_access() {
        let mut v = V::new(1.0, 2.0);
        v.x = 10.0;
        v.y = 20.0;
        assert_eq!(v, V::new(10.0, 20.0));
    }

    #[test]
    fn v2_lerp() {
        let a = V::new(0.0, 0.0);
        let b = V::new(100.0, 200.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x - 50.0).abs() < 1e-10);
        assert!((mid.y - 100.0).abs() < 1e-10);
    }

    #[test]
    fn v2_distance() {
        let a = V::new(0.0, 0.0);
        let b = V::new(3.0, 4.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn v2_area() {
        assert!((V::new(10.0, 20.0).area() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn v2_arithmetic() {
        let a = V::new(1.0, 2.0);
        let b = V::new(3.0, 4.0);
        assert_eq!(a + b, V::new(4.0, 6.0));
        assert_eq!(b - a, V::new(2.0, 2.0));
        assert_eq!(-a, V::new(-1.0, -2.0));
        assert_eq!(a * 3.0, V::new(3.0, 6.0));
        assert_eq!(2.0 * a, V::new(2.0, 4.0));
        assert_eq!(a / 2.0, V::new(0.5, 1.0));
    }

    #[test]
    fn v3_operations() {
        let a = V([1.0, 2.0, 3.0]);
        let b = V([4.0, 5.0, 6.0]);
        assert_eq!(a + b, V([5.0, 7.0, 9.0]));
        assert_eq!(a.x, 1.0);
        assert_eq!(a.z, 3.0);
        assert!((a.dot(b) - 32.0).abs() < 1e-10);
    }

    #[test]
    fn v_generic_n() {
        let v: V<8> = V::splat(2.0);
        assert_eq!(v.component_sum(), 16.0);
        assert_eq!(v.product(), 256.0);
        let big: V<128> = V::ZERO;
        assert_eq!(big.component_sum(), 0.0);
    }

    #[test]
    fn v2_from_conversions() {
        assert_eq!(Point::from((3.0, 4.0)), Point::new(3.0, 4.0));
        assert_eq!(Point::from((3i32, 4i32)), Point::new(3.0, 4.0));
        assert_eq!(Point::from([3.0, 4.0]), Point::new(3.0, 4.0));
        assert_eq!(Point::from(5.0), Point::new(5.0, 5.0));
        let (x, y): (f64, f64) = Point::new(1.0, 2.0).into();
        assert_eq!((x, y), (1.0, 2.0));
        let arr: [f64; 2] = Point::new(1.0, 2.0).into();
        assert_eq!(arr, [1.0, 2.0]);
        assert_eq!(Size::from((10.0, 20.0)), Size::new(10.0, 20.0));
        assert_eq!(Size::from((100u32, 200u32)), Size::new(100.0, 200.0));
    }

    #[test]
    fn point_size_same_type() {
        let p = Point::new(3.0, 4.0);
        let s: Size = p;
        assert_eq!(s, Size::new(3.0, 4.0));
    }

    #[test]
    fn region_contains() {
        let r = Rect::new(10.0, 10.0, 100.0, 50.0);
        assert!(r.contains(V::new(50.0, 30.0)));
        assert!(!r.contains(V::new(5.0, 30.0)));
        assert!(!r.contains(V::new(50.0, 70.0)));
    }

    #[test]
    fn region_accessors() {
        let r = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(r.x(), 10.0);
        assert_eq!(r.y(), 20.0);
        assert_eq!(r.w(), 100.0);
        assert_eq!(r.h(), 50.0);
        assert_eq!(r.right(), 110.0);
        assert_eq!(r.bottom(), 70.0);
        let c = r.center();
        assert!((c.x - 60.0).abs() < 1e-10);
        assert!((c.y - 45.0).abs() < 1e-10);
    }

    #[test]
    fn region_lerp() {
        let a = Rect::new(0.0, 0.0, 100.0, 50.0);
        let b = Rect::new(100.0, 100.0, 200.0, 100.0);
        let mid = a.lerp(b, 0.5);
        assert!((mid.x() - 50.0).abs() < 1e-10);
        assert!((mid.w() - 150.0).abs() < 1e-10);
    }

    #[test]
    fn region_from_parts() {
        let r = Rect::from_parts(V::new(5.0, 10.0), V::new(20.0, 30.0));
        assert_eq!(r.x(), 5.0);
        assert_eq!(r.h(), 30.0);
    }

    #[test]
    fn region_from_conversions() {
        assert_eq!(
            Rect::from((1.0, 2.0, 3.0, 4.0)),
            Rect::new(1.0, 2.0, 3.0, 4.0)
        );
        assert_eq!(
            Rect::from([1.0, 2.0, 3.0, 4.0]),
            Rect::new(1.0, 2.0, 3.0, 4.0)
        );
        let r: Rect = (V::new(1.0, 2.0), V::new(3.0, 4.0)).into();
        assert_eq!(r, Rect::new(1.0, 2.0, 3.0, 4.0));
        let r: Rect = V::new(800.0, 600.0).into();
        assert_eq!(r, Rect::new(0.0, 0.0, 800.0, 600.0));
    }

    #[test]
    fn region_3d() {
        let r = Region::<3>::from_parts(V([0.0, 0.0, 0.0]), V([10.0, 20.0, 30.0]));
        assert!(r.contains(V([5.0, 10.0, 15.0])));
        assert!(!r.contains(V([15.0, 10.0, 15.0])));
        assert_eq!(r.center(), V([5.0, 10.0, 15.0]));
    }

    #[test]
    fn constraints_clamp() {
        let c = Constraints {
            min: V::new(10.0, 10.0),
            max: V::new(200.0, 200.0),
        };
        assert_eq!(c.clamp(V::new(5.0, 300.0)), V::new(10.0, 200.0));
    }

    #[test]
    fn scroll_visible_range() {
        let s = ScrollState {
            offset: V::new(0.0, 280.0),
        };
        let range = s.visible_range(28.0, 280.0, 1000);
        assert_eq!(range.start, 10);
        assert!(range.end <= 21);
    }

    #[test]
    fn scroll_visible_range_edge_cases() {
        let s = ScrollState { offset: V::ZERO };
        assert_eq!(s.visible_range(0.0, 100.0, 100), 0..0);
        assert_eq!(s.visible_range(10.0, 100.0, 0), 0..0);
    }

    // ── Matrix tests ────────────────────────────────────────────────

    #[test]
    fn mat_identity() {
        let id = Mat4::identity();
        let v = V([1.0, 2.0, 3.0, 4.0]);
        assert_eq!(id * v, v);
    }

    #[test]
    fn mat_transpose() {
        let m = Matrix::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let t = m.transpose();
        assert_eq!(t.data, [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]);
    }

    #[test]
    fn mat_mul_vec() {
        let m = Mat3::new([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
        let v = V([1.0, 1.0, 1.0]);
        assert_eq!(m * v, V([1.0, 2.0, 3.0]));
    }

    #[test]
    fn mat_add_sub() {
        let a = Mat2::new([[1.0, 2.0], [3.0, 4.0]]);
        let b = Mat2::new([[10.0, 20.0], [30.0, 40.0]]);
        let sum = a + b;
        assert_eq!(sum.data, [[11.0, 22.0], [33.0, 44.0]]);
        let diff = b - a;
        assert_eq!(diff.data, [[9.0, 18.0], [27.0, 36.0]]);
    }

    #[test]
    fn mat4_transform_point() {
        let t = Mat4::translation(V([10.0, 20.0, 30.0]));
        let p = t.transform_point(V([1.0, 2.0, 3.0]));
        assert!((p.0[0] - 11.0).abs() < 1e-10);
        assert!((p.0[1] - 22.0).abs() < 1e-10);
        assert!((p.0[2] - 33.0).abs() < 1e-10);
    }

    #[test]
    fn mat4_transform_dir() {
        let t = Mat4::translation(V([100.0, 200.0, 300.0]));
        let d = t.transform_dir(V([1.0, 0.0, 0.0]));
        // Direction ignores translation
        assert!((d.0[0] - 1.0).abs() < 1e-10);
        assert!((d.0[1]).abs() < 1e-10);
    }

    #[test]
    fn mat_scaling() {
        let s = Mat4::scaling(2.0);
        let p = s.transform_point(V([1.0, 2.0, 3.0]));
        assert!((p.0[0] - 2.0).abs() < 1e-10);
        assert!((p.0[1] - 4.0).abs() < 1e-10);
        assert!((p.0[2] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn mat_v_roundtrip() {
        let v = V([3.0, 4.0, 5.0]);
        let col: Matrix<3, 1> = v.into();
        let back: V<3> = col.into();
        assert_eq!(back, v);
    }

    #[test]
    fn mat_row_col() {
        let m = Mat3::new([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        assert_eq!(m.row(1), V([4.0, 5.0, 6.0]));
        assert_eq!(m.col(0), V([1.0, 4.0, 7.0]));
    }
}
