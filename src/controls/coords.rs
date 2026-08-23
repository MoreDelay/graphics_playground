//! Module to transform global locations to viewport-local locations

use iced::wgpu;
use iced_winit::winit::dpi::{PhysicalInsets, PhysicalSize};
use nalgebra as na;

/// Transforms
#[derive(Debug, Clone, Copy)]
pub struct LocalCoords {
    /// Physical location inside the window
    bounds: PhysicalInsets<u32>,
    /// Scale factor between logical and physical coordinates
    scale_factor: f32,
}

impl LocalCoords {
    /// Create new local coordinate transform
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
    pub const fn local_vector(&self, vector: na::Vector2<f32>) -> LocalVector {
        LocalVector::wrap(vector)
    }

    /// Create a local point within the handled area
    pub fn local_point(&self, point: na::Point2<f32>) -> Option<LocalPoint> {
        let PhysicalInsets {
            top,
            left,
            bottom,
            right,
        } = self.bounds.cast::<f32>();

        let inside = (left <= point.x && point.x <= right - 1.)
            && (top <= point.y && point.y <= bottom - 1.);

        let point = point - *self.offset();
        inside.then_some(LocalPoint::wrap(point))
    }

    /// Get the offset of the local origin in global (physical) coordinates
    const fn offset(&self) -> LocalVector {
        let bounds = self.bounds;
        let offset = na::Vector2::new(bounds.left as f32, bounds.top as f32);
        LocalVector::wrap(offset)
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

/// Wrapper to mark coordinates to be in some local coordinates
#[derive(Debug, Clone, Copy)]
pub struct Local<T>(pub Physical<T>)
where
    T: std::fmt::Debug + Clone + Copy;

impl<T> std::ops::Deref for Local<T>
where
    T: std::fmt::Debug + Clone + Copy,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0.0
    }
}
impl<T> std::ops::DerefMut for Local<T>
where
    T: std::fmt::Debug + Clone + Copy,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.0
    }
}

impl<T> Local<T>
where
    T: std::fmt::Debug + Clone + Copy,
{
    /// Wrap a type to mark them as local
    pub const fn wrap(inner: T) -> Self {
        Self(Physical(inner))
    }
}

/// Helper alias for a local point
pub type LocalPoint = Local<na::Point2<f32>>;
/// Helper alias for a local vector
pub type LocalVector = Local<na::Vector2<f32>>;
