use iced::wgpu;
use iced_winit::winit::dpi::{PhysicalInsets, PhysicalSize};
use nalgebra as na;

#[derive(Debug, Clone, Copy)]
pub struct LocalCoords {
    bounds: PhysicalInsets<u32>,
    scale_factor: f32,
}

impl LocalCoords {
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

    pub const fn resized(self, bounds: PhysicalInsets<u32>) -> Self {
        Self { bounds, ..self }
    }

    pub const fn scale_changed(self, scale_factor: f32) -> Self {
        Self {
            scale_factor,
            ..self
        }
    }

    pub const fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize {
            width: self.bounds.right - self.bounds.left,
            height: self.bounds.bottom - self.bounds.top,
        }
    }

    pub const fn bounds(self) -> PhysicalInsets<u32> {
        self.bounds
    }

    #[expect(dead_code)]
    pub const fn scale_factor(self) -> f32 {
        self.scale_factor
    }

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

    pub fn local_vector(&self, vector: iced::Vector) -> LocalVector {
        let iced::Vector { x, y } = vector;
        let vector = na::Vector2::new(x, y) / self.scale_factor;
        LocalVector::wrap(vector)
    }

    pub fn local_point(&self, point: iced::Point) -> Option<LocalPoint> {
        let iced::Point { x, y } = point;
        let PhysicalInsets {
            top,
            left,
            bottom,
            right,
        } = self.bounds.cast::<f32>();

        let inside = (left <= x && x <= right - 1.) && (top <= y && y <= bottom - 1.);

        let point = na::Point2::new(x, y) / self.scale_factor;
        let point = point - *self.offset();
        inside.then_some(LocalPoint::wrap(point))
    }

    const fn offset(&self) -> LocalVector {
        let bounds = self.bounds;
        #[expect(clippy::cast_precision_loss)]
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

#[derive(Debug, Clone, Copy)]
pub struct Physical<T>(pub T)
where
    T: std::fmt::Debug + Clone + Copy;

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
    pub const fn wrap(inner: T) -> Self {
        Self(Physical(inner))
    }
}

pub type LocalPoint = Local<na::Point2<f32>>;
pub type LocalVector = Local<na::Vector2<f32>>;
