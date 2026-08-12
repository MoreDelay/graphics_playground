use iced::wgpu;
use iced_winit::winit::dpi::{PhysicalInsets, PhysicalSize};

use crate::controls::coords::LocalCoords;
use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};

pub struct Viewport {
    coords: LocalCoords,
}

impl Viewport {
    pub fn new() -> Self {
        let coords = LocalCoords::default();
        Self { coords }
    }

    pub fn update_coords(&mut self, coords: LocalCoords) -> bool {
        let equal = self.coords == coords;
        self.coords = coords;
        !equal
    }

    pub fn update_bounds(&mut self, bounds: PhysicalInsets<u32>) -> bool {
        let coords = self.coords.resized(bounds);
        self.update_coords(coords)
    }

    pub fn update_scale_factor(&mut self, scale_factor: f32) {
        let coords = self.coords.scale_changed(scale_factor);
        self.update_coords(coords);
    }

    pub const fn coords(&self) -> LocalCoords {
        self.coords
    }

    pub const fn size(&self) -> PhysicalSize<u32> {
        self.coords.size()
    }

    pub const fn extent(&self) -> Option<wgpu::Extent3d> {
        self.coords.extent()
    }

    pub fn draw(
        &self,
        passthru: &PassThruPipeline,
        encoder: &mut wgpu::CommandEncoder,
        rendering: &PassThruTexture,
        target: &wgpu::TextureView,
    ) {
        let got_size = rendering.texture().size();
        let expected_size = self.coords.extent().expect("viewport has no viewing area");
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
        } = self.coords.bounds().cast();

        let bounds = iced::Rectangle {
            x: left,
            y: top,
            width: right - left,
            height: bottom - top,
        };

        // limit rendering to the viewport bounds
        pass.set_viewport(bounds.x, bounds.y, bounds.width, bounds.height, 0., 1.);

        passthru.draw(&mut pass, rendering);
    }
}
