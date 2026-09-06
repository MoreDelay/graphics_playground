//! Handling of rendering to a specified viewport within the window

use iced::wgpu;
use iced_winit::winit::dpi::{PhysicalInsets, PhysicalSize};
use nalgebra as na;

use crate::controls::coords::LocalCoords;
use crate::instruments::bind::image::ViewportRaw;
use crate::instruments::mesh::InstanceBuffer;
use crate::instruments::mesh::primitives::InstanceRaw;
use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};

/// A region on the window to which a widget's content is rendered to
pub struct ViewportGui {
    /// The local coordinate system of the viewport
    coords: LocalCoords,
    /// Quad mesh that provides the vertices
    quad_instance: Option<InstanceBuffer<InstanceRaw>>,
}

impl ViewportGui {
    /// Create a new viewport with no size
    pub fn new() -> Self {
        let coords = LocalCoords::default();
        let quad_instance = None;
        Self {
            coords,
            quad_instance,
        }
    }

    /// Update to a new local coordinate system
    pub fn update_coords(&mut self, coords: LocalCoords) -> bool {
        let equal = self.coords == coords;
        if !equal {
            self.coords = coords;
            self.quad_instance = None;
        }
        !equal
    }

    /// Resize the local coordinates
    pub fn update_bounds(&mut self, bounds: PhysicalInsets<u32>) -> bool {
        let coords = self.coords.resized(bounds);
        self.update_coords(coords)
    }

    /// Change the scale factor
    pub fn update_scale_factor(&mut self, scale_factor: f32) {
        let coords = self.coords.scale_changed(scale_factor);
        self.update_coords(coords);
    }

    /// Get access to the local coordinate transform
    pub const fn coords(&self) -> LocalCoords {
        self.coords
    }

    /// Get the size of this viewport
    pub const fn size(&self) -> PhysicalSize<u32> {
        self.coords.size()
    }

    /// Get the size of this viewport to be used with wgpu
    pub const fn extent(&self) -> Option<wgpu::Extent3d> {
        self.coords.extent()
    }

    /// Draw to this viewport
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

        let PhysicalSize { width, height } = self.coords.size();
        let area = width * height;
        assert!(area > 0, "can not render to a viewport with no area");

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

/// Handles the viewport vertex transformation
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct ScrollableViewportState {
    /// Default zoom level uses this many pixels to display a length of one
    pixel_per_unit: f32,
    /// Total size that can be scrolled to (disregarding effects of zoom)
    area: na::Vector2<f32>,
    /// The physical size that is displayed in the view
    view: PhysicalSize<u32>,
    /// The viewport has moved away from the image center by this amount
    offset: na::Point2<f32>,
    /// Image is scaled by this factor
    zoom: f32,
}

impl ScrollableViewportState {
    /// The limit for magnification
    const ZOOM_MAX: f32 = 100.0;
    /// The limit for minification
    const ZOOM_MIN: f32 = 0.05;

    /// Create a new scrollable viewport centered in the middle and with neutral zoom
    pub const fn new(area: na::Vector2<f32>, view: PhysicalSize<u32>, pixel_per_unit: f32) -> Self {
        Self {
            pixel_per_unit,
            area,
            view,
            offset: na::Point2::new(0., 0.),
            zoom: 1.,
        }
    }

    /// Update the size of the total scrollable area
    pub fn resize_area(&mut self, area: na::Vector2<f32>) -> bool {
        self.reset();

        let before = std::mem::replace(&mut self.area, area);
        before != self.area
    }

    /// Update the size of the visible area
    pub fn resize_view(&mut self, view: PhysicalSize<u32>) -> bool {
        let before = std::mem::replace(&mut self.view, view);
        before != self.view
    }

    /// Reset the pose to be centered in the middle and with neutral zoom
    pub const fn reset_position(&mut self) {
        self.offset = na::Point2::new(0., 0.);
    }

    /// Reset the zoom to neutral
    pub const fn reset_zoom(&mut self) {
        self.zoom = 1.;
    }

    /// Reset the pose back to the default
    pub const fn reset(&mut self) {
        self.reset_position();
        self.reset_zoom();
    }

    /// Scroll the contents of this viewport in the given direction
    pub fn scroll(&mut self, pan_vector: na::Vector2<f32>) {
        self.offset -= pan_vector / self.zoom;
        self.clamp_offset();
    }

    /// Get the size of scrollable area
    pub const fn area(&self) -> na::Vector2<f32> {
        self.area
    }

    /// Get the size of the viewable area
    pub const fn size(&self) -> PhysicalSize<u32> {
        self.view
    }

    /// Get the current zoom level
    pub const fn zoom(&self) -> f32 {
        self.zoom
    }

    /// Update the zoom level
    pub fn set_zoom(&mut self, zoom: f32, fix_point: na::Point2<f32>) {
        let zoom = zoom.clamp(Self::ZOOM_MIN, Self::ZOOM_MAX);
        // Changing the zoom factor without adapting the offset will keep the viewport center
        // unmoving. Instead, the fix-point should be unmoving, which is an offset from the viewport
        // center.
        let offset = self.offset + fix_point.coords * (1. / self.zoom - 1. / zoom);

        self.offset = offset;
        self.zoom = zoom;

        // when the image is at the border, it might move out of frame by zooming
        self.clamp_offset();
    }

    /// Make sure that at least 10% of the viewport area shows part of the image.
    fn clamp_offset(&mut self) {
        const FILLED_MINIMUM: f32 = 0.1;

        let area = self.area;

        let view = self.view.cast::<f32>();
        let view = na::Vector2::new(view.width, view.height);

        let limit = area / 2. + view / 2. - view * FILLED_MINIMUM;
        let offset = self.offset;
        let x = offset.x.clamp(-limit.x, limit.x);
        let y = offset.y.clamp(-limit.y, limit.y);
        let clamped = na::Point2::new(x, y);

        self.offset = clamped;
    }

    /// Create the [`ViewportRaw`] corresponding to the current parameters
    pub fn as_raw(&self) -> ViewportRaw {
        let PhysicalSize { width, height } = self.view.cast::<f32>();
        let factor = 2. * self.pixel_per_unit * self.zoom;
        let sx = factor / width;
        let sy = factor / height;
        let offset = na::Point2::new(-self.offset.x * sx, -self.offset.y * sy);
        let view0 = na::Vector3::new(sx, 0., 0.);
        let view1 = na::Vector3::new(0., sy, 0.);
        let view2 = offset.to_homogeneous();

        ViewportRaw {
            view0: view0.into(),
            view1: view1.into(),
            view2: view2.into(),
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        }
    }
}
