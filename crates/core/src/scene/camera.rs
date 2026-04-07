use crate::layout::{V, V3};

// ═══════════════════════════════════════════════════════════════════════════
// ── Camera ───────────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════

/// Projection mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Projection {
    /// Field of view in radians, aspect ratio.
    Perspective {
        fov: f64,
        aspect: f64,
        near: f64,
        far: f64,
    },
    /// Orthographic half-extents.
    Orthographic {
        width: f64,
        height: f64,
        near: f64,
        far: f64,
    },
}

/// Camera — position, target, up + projection.
#[derive(Debug, Clone)]
pub struct Camera {
    pub eye: V3,
    pub target: V3,
    pub up: V3,
    pub projection: Projection,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            eye: V([0.0, 0.0, 5.0]),
            target: V3::ZERO,
            up: V([0.0, 1.0, 0.0]),
            projection: Projection::Perspective {
                fov: std::f64::consts::FRAC_PI_4,
                aspect: 16.0 / 9.0,
                near: 0.1,
                far: 1000.0,
            },
        }
    }
}

impl Camera {
    /// View direction (normalized).
    pub fn forward(&self) -> V3 {
        (self.target - self.eye).normalized()
    }

    /// Right vector (normalized).
    pub fn right(&self) -> V3 {
        self.forward().cross(self.up).normalized()
    }

    /// Up vector (true orthonormal up, not the hint).
    pub fn true_up(&self) -> V3 {
        self.right().cross(self.forward()).normalized()
    }

    /// Project a world-space point to normalized device coordinates (-1..1, -1..1, depth).
    /// Returns None if behind the camera near plane.
    pub fn project(&self, p: V3) -> Option<V3> {
        let f = self.forward();
        let r = self.right();
        let u = self.true_up();

        // View-space position
        let rel = p - self.eye;
        let vx = rel.dot(r);
        let vy = rel.dot(u);
        let vz = rel.dot(f);

        match self.projection {
            Projection::Perspective {
                fov,
                aspect,
                near,
                far,
            } => {
                if vz < near || vz > far {
                    return None;
                }
                let half_h = (fov / 2.0).tan();
                let half_w = half_h * aspect;
                let ndc_x = vx / (vz * half_w);
                let ndc_y = vy / (vz * half_h);
                let ndc_z = (vz - near) / (far - near);
                Some(V([ndc_x, ndc_y, ndc_z]))
            }
            Projection::Orthographic {
                width,
                height,
                near,
                far,
            } => {
                if vz < near || vz > far {
                    return None;
                }
                let ndc_x = vx / (width / 2.0);
                let ndc_y = vy / (height / 2.0);
                let ndc_z = (vz - near) / (far - near);
                Some(V([ndc_x, ndc_y, ndc_z]))
            }
        }
    }

    /// Project to screen coordinates (pixel space).
    /// Returns (screen_x, screen_y, depth) or None if behind camera.
    pub fn project_to_screen(
        &self,
        p: V3,
        screen_w: f64,
        screen_h: f64,
    ) -> Option<(f64, f64, f64)> {
        self.project(p).map(|ndc| {
            let sx = (ndc.0[0] * 0.5 + 0.5) * screen_w;
            let sy = (1.0 - (ndc.0[1] * 0.5 + 0.5)) * screen_h;
            (sx, sy, ndc.0[2])
        })
    }

    /// Orbit around the target point by horizontal/vertical angles (radians).
    pub fn orbit(&mut self, d_azimuth: f64, d_elevation: f64) {
        let dist = (self.eye - self.target).magnitude();
        let f = self.forward();
        let r = self.right();
        let u = self.true_up();

        // Rotate the forward direction by azimuth (around true-up)
        let cos_a = d_azimuth.cos();
        let sin_a = d_azimuth.sin();
        let horiz = f * cos_a + r * sin_a;

        // Tilt by elevation (toward true-up)
        let cos_e = d_elevation.cos();
        let sin_e = d_elevation.sin();
        let tilted = (horiz * cos_e + u * sin_e).normalized();

        // Clamp: reject elevation if it would place the camera at a pole
        // (forward ≈ ±world-up → degenerate right vector next frame).
        let new_fwd = if tilted.dot(self.up).abs() > 0.99 {
            horiz.normalized()
        } else {
            tilted
        };

        let new_eye = self.target - new_fwd * dist;
        self.eye = new_eye;
    }

    /// Zoom by moving eye along the view direction.
    pub fn zoom(&mut self, factor: f64) {
        let offset = self.eye - self.target;
        let dist = (offset.magnitude() * factor).max(0.1);
        self.eye = self.target + offset.normalized() * dist;
    }

    /// Move the camera forward/backward along the view direction (FPS-style).
    pub fn fly(&mut self, amount: f64) {
        let fwd = self.forward() * amount;
        self.eye = self.eye + fwd;
        self.target = self.target + fwd;
    }

    /// Strafe left/right (perpendicular to view direction on the ground plane).
    pub fn strafe(&mut self, amount: f64) {
        let r = self.right() * amount;
        self.eye = self.eye + r;
        self.target = self.target + r;
    }

    /// Move the camera up/down along the world up axis.
    pub fn elevate(&mut self, amount: f64) {
        let u = self.up * amount;
        self.eye = self.eye + u;
        self.target = self.target + u;
    }

    /// Create a [`CameraView`] — binds this camera to a screen size so it
    /// can be used as a [`Viewport`](crate::render::Viewport).
    pub fn view(&self, screen_w: f64, screen_h: f64) -> CameraView<'_> {
        CameraView {
            camera: self,
            screen_w,
            screen_h,
        }
    }
}

/// Camera + screen dimensions, implementing [`Viewport`](crate::render::Viewport).
pub struct CameraView<'a> {
    pub camera: &'a Camera,
    pub screen_w: f64,
    pub screen_h: f64,
}

impl crate::render::Viewport<V3> for CameraView<'_> {
    fn project(&self, point: V3) -> Option<(f64, f64, f64)> {
        self.camera
            .project_to_screen(point, self.screen_w, self.screen_h)
    }

    fn screen_bounds(&self) -> crate::layout::Rect {
        crate::layout::Rect::new(0.0, 0.0, self.screen_w, self.screen_h)
    }
}

