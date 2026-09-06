//! Module to transform global locations to viewport-local locations

use iced::wgpu;
use iced_winit::winit::dpi::{PhysicalInsets, PhysicalSize};
use nalgebra as na;

/// Defines a local coordinate system
///
/// The positive axes point to the right and up, respectively for x- and y-axis.
#[derive(Debug, Clone, Copy)]
pub struct LocalCoords {
    /// Physical location inside the window
    ///
    /// The origin is at the center of these bounds
    bounds: PhysicalInsets<u32>,
    /// Scale factor between logical and physical coordinates
    scale_factor: f32,
}

impl LocalCoords {
    /// Create new local coordinate transform
    ///
    /// The origin is at the center of the provided bounds.
    pub const fn new(bounds: PhysicalInsets<u32>, scale_factor: f32) -> Self {
        assert!(
            scale_factor > 0.,
            "non-positive scale factor makes no sense"
        );
        Self {
            bounds,
            scale_factor,
        }
    }

    /// Create new local coordinates with updated sizes
    pub const fn resized(self, bounds: PhysicalInsets<u32>) -> Self {
        Self { bounds, ..self }
    }

    /// Create new local coordinates with updated scale factor
    pub const fn scale_changed(self, scale_factor: f32) -> Self {
        Self {
            scale_factor,
            ..self
        }
    }

    /// Get the physical size of the area handled by this
    pub const fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize {
            width: self.bounds.right - self.bounds.left,
            height: self.bounds.bottom - self.bounds.top,
        }
    }

    /// Get the physical bounds of the area handled by this
    pub const fn bounds(self) -> PhysicalInsets<u32> {
        self.bounds
    }

    /// Get the scale factor
    pub const fn scale_factor(self) -> f32 {
        self.scale_factor
    }

    /// Get the bounds for use with wgpu
    pub const fn extent(&self) -> Option<wgpu::Extent3d> {
        let size = self.size();
        let has_area = size.width > 0 && size.height > 0;
        if !has_area {
            return None;
        }
        // confirmed with a checkerboard image that this is the physical size of the viewport
        let extent = wgpu::Extent3d {
            width: self.bounds.right - self.bounds.left,
            height: self.bounds.bottom - self.bounds.top,
            depth_or_array_layers: 1,
        };
        Some(extent)
    }

    /// Create a local vector
    #[expect(clippy::unused_self)]
    pub fn local_vector(&self, vector: Physical<na::Vector2<f32>>) -> na::Vector2<f32> {
        let v = vector.0;
        na::Vector2::new(v.x, -v.y)
    }

    /// Create a local point within the handled area
    pub fn local_point(&self, point: Physical<na::Point2<f32>>) -> Option<na::Point2<f32>> {
        let PhysicalInsets {
            top,
            left,
            bottom,
            right,
        } = self.bounds.cast::<f32>();

        let p = point.0;

        let inside = (left <= p.x && p.x <= right - 1.) && (top <= p.y && p.y <= bottom - 1.);
        if !inside {
            return None;
        }

        let p = p - self.offset();
        // flip y axis as window coordinates has this pointing down, we want it up
        let p = na::Point2::new(p.x, -p.y);
        Some(p)
    }

    /// Get the offset of the local origin in global (physical) coordinates
    fn offset(&self) -> na::Vector2<f32> {
        let bounds = self.bounds.cast();
        let corner = na::Vector2::new(bounds.left, bounds.top);
        let half = self.size().cast();
        let half = na::Vector2::new(half.width, half.height) / 2.;
        corner + half
    }
}

impl PartialEq for LocalCoords {
    fn eq(&self, other: &Self) -> bool {
        // Ignore scale factor for equality check, because scale factor does not influence physical
        // coordinates.
        self.bounds == other.bounds
    }
}

impl Default for LocalCoords {
    fn default() -> Self {
        Self {
            bounds: PhysicalInsets::default(),
            scale_factor: 1.,
        }
    }
}

/// Wrapper to mark coordinates to be in physical coordinates
#[derive(Debug, Clone, Copy)]
pub struct Physical<T>(pub T)
where
    T: std::fmt::Debug + Clone + Copy;
