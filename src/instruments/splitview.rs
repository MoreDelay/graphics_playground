//! Contains the functionality to draw two images in a split view

use iced::wgpu;

use crate::image::ClampedSplit;
use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};
use crate::viewport::ViewportGui;

/// Draw two images to the viewport, split as specified by [`ClampedSplit`]
pub fn draw_splitted(
    passthru: &PassThruPipeline,
    encoder: &mut wgpu::CommandEncoder,
    viewport: &ViewportGui,
    left: &PassThruTexture,
    right: &PassThruTexture,
    target: &wgpu::TextureView,
    split: ClampedSplit,
) {
    let got_size_left = left.texture().size();
    let got_size_right = right.texture().size();
    let size = viewport.extent().expect("viewport has no viewing area");
    assert_eq!(
        got_size_left, size,
        "left texture does not match viewport size"
    );
    assert_eq!(
        got_size_right, size,
        "right texture does not match viewport size"
    );

    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("Image Split Render Pass"),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    let wgpu::Extent3d { width, height, .. } = size;

    let full = iced::Rectangle {
        x: 0,
        y: 0,
        width,
        height,
    };

    let (bounds_left, bounds_right) = match split {
        ClampedSplit::FullLeft => (None, Some(full)),
        ClampedSplit::Split(split) => {
            let split = split + viewport.size().width as f32 / 2.;
            let split = split as u32;
            let left = iced::Rectangle {
                x: 0,
                y: 0,
                width: split.saturating_sub(1),
                height,
            };
            let x = width.min(split + 1);
            let right = iced::Rectangle {
                x,
                y: 0,
                width: width.saturating_sub(x),
                height,
            };
            (Some(left), Some(right))
        }
        ClampedSplit::FullRight => (Some(full), None),
    };

    if let Some(bounds) = bounds_left {
        pass.set_scissor_rect(bounds.x, bounds.y, bounds.width, bounds.height);
        passthru.draw(&mut pass, left);
    }
    if let Some(bounds) = bounds_right {
        pass.set_scissor_rect(bounds.x, bounds.y, bounds.width, bounds.height);
        passthru.draw(&mut pass, right);
    }
}
