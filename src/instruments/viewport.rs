use iced::wgpu;
use iced_winit::winit::dpi::{PhysicalInsets, PhysicalSize};
use nalgebra as na;

use crate::instruments::GpuContext;
use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};

pub struct Viewport {
    pipeline: PassThruPipeline,
    bounds: Option<PhysicalInsets<u32>>,
    scale_factor: f32,
}

impl Viewport {
    pub fn new(ctx: &GpuContext, output_format: wgpu::TextureFormat) -> Self {
        let pipeline = PassThruPipeline::new(ctx, output_format);
        Self {
            pipeline,
            bounds: None,
            scale_factor: 1.,
        }
    }

    pub fn resize(&mut self, bounds: PhysicalInsets<u32>) -> bool {
        let PhysicalInsets {
            top,
            left,
            bottom,
            right,
        } = bounds;

        let width = right - left;
        let height = bottom - top;
        let last = if width == 0 || height == 0 {
            self.bounds.take()
        } else {
            self.bounds.replace(bounds)
        };
        last != self.bounds
    }

    pub const fn set_scale_factor(&mut self, scale_factor: f32) {
        self.scale_factor = scale_factor;
    }

    pub fn create_texture(&self, ctx: &GpuContext) -> Option<PassThruTexture> {
        let size = bounds_to_extent(self.bounds?);
        let texture = self.pipeline.create_texture(ctx, size);
        Some(texture)
    }

    pub fn draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rendering: &PassThruTexture,
        target: &wgpu::TextureView,
    ) {
        let got_size = rendering.texture().size();
        let bounds = self
            .bounds
            .expect("only call draw when viewport has actual area");
        let expected_size = bounds_to_extent(bounds);
        assert_eq!(
            got_size, expected_size,
            "texture does not match viewport size"
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Viewport PassThru Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // iced drew the gui already, so load that
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        let PhysicalInsets {
            top,
            left,
            bottom,
            right,
        } = bounds.cast();

        let bounds = iced::Rectangle {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        };

        // limit rendering to the viewport bounds
        pass.set_viewport(bounds.x, bounds.y, bounds.width, bounds.height, 0., 1.);

        self.pipeline.draw(&mut pass, rendering);
    }

    pub fn size(&self) -> Option<PhysicalSize<u32>> {
        let bounds = self.bounds?;
        Some(PhysicalSize {
            width: bounds.right - bounds.left,
            height: bounds.bottom - bounds.top,
        })
    }

    pub fn extent(&self) -> Option<wgpu::Extent3d> {
        self.bounds.map(bounds_to_extent)
    }

    pub fn to_vp_vector(&self, vector: iced::Vector) -> VPVector {
        let iced::Vector { x, y } = vector;
        let vector = na::Vector2::new(x, y) / self.scale_factor;
        VPVector::wrap(vector)
    }

    pub fn to_vp_point(&self, point: iced::Point) -> VPPoint {
        let iced::Point { x, y } = point;
        let point = na::Point2::new(x, y) / self.scale_factor;
        let point = point - *self.offset();
        VPPoint::wrap(point)
    }

    fn offset(&self) -> VPVector {
        let bounds = self.bounds.unwrap_or_default();
        #[expect(clippy::cast_precision_loss)]
        let offset = na::Vector2::new(bounds.left as f32, bounds.top as f32);
        VPVector::wrap(offset)
    }
}

const fn bounds_to_extent(bounds: PhysicalInsets<u32>) -> wgpu::Extent3d {
    let PhysicalInsets {
        top,
        left,
        bottom,
        right,
    } = bounds;
    // confirmed with a checkerboard image that this is the physical size of the viewport
    wgpu::Extent3d {
        width: right - left,
        height: bottom - top,
        depth_or_array_layers: 1,
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Physical<T>(pub T)
where
    T: std::fmt::Debug + Clone + Copy;

#[derive(Debug, Clone, Copy)]
pub struct VP<T>(pub Physical<T>)
where
    T: std::fmt::Debug + Clone + Copy;

impl<T> std::ops::Deref for VP<T>
where
    T: std::fmt::Debug + Clone + Copy,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0.0
    }
}
impl<T> std::ops::DerefMut for VP<T>
where
    T: std::fmt::Debug + Clone + Copy,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.0
    }
}

impl<T> VP<T>
where
    T: std::fmt::Debug + Clone + Copy,
{
    pub const fn wrap(inner: T) -> Self {
        Self(Physical(inner))
    }
}

pub type VPPoint = VP<na::Point2<f32>>;
pub type VPVector = VP<na::Vector2<f32>>;
