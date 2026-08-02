use iced::wgpu;
use iced_winit::winit::dpi::{PhysicalInsets, PhysicalSize};
use nalgebra as na;

use crate::instruments::GpuContext;
use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};

pub struct Viewport {
    pipeline: PassThruPipeline,
    bounds: Option<PhysicalInsets<u32>>,
}

impl Viewport {
    pub fn new(ctx: &GpuContext, output_format: wgpu::TextureFormat) -> Self {
        let pipeline = PassThruPipeline::new(ctx, output_format);
        let bounds = None;
        Self { pipeline, bounds }
    }

    pub const fn resize(&mut self, bounds: PhysicalInsets<u32>) {
        let PhysicalInsets {
            top,
            left,
            bottom,
            right,
        } = bounds;

        let width = right - left;
        let height = bottom - top;
        if width == 0 || height == 0 {
            self.bounds = None;
        } else {
            self.bounds = Some(bounds);
        }
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
