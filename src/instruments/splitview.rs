use iced::wgpu;

use crate::image::ComparisonSplit;
use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};
use crate::instruments::viewport::Viewport;

pub fn draw_splitted(
    passthru: &PassThruPipeline,
    encoder: &mut wgpu::CommandEncoder,
    viewport: &Viewport,
    left: &PassThruTexture,
    right: &PassThruTexture,
    target: &wgpu::TextureView,
    split: ComparisonSplit,
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
                load: wgpu::LoadOp::Load, // iced drew the gui already, so load that
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });

    let wgpu::Extent3d { width, height, .. } = size;
    let width = width as f32;
    let height = height as f32;

    let full = iced::Rectangle {
        x: 0.,
        y: 0.,
        width,
        height,
    };

    let (bounds_left, bounds_right) = match split {
        ComparisonSplit::FullLeft => (Some(full), None),
        ComparisonSplit::Split(split) => {
            let split = split as f32;
            let left = iced::Rectangle {
                x: 0.,
                y: 0.,
                width: split - 1.,
                height,
            };
            let right = iced::Rectangle {
                x: split + 1.,
                y: 0.,
                width: width - (split + 1.),
                height,
            };
            (Some(left), Some(right))
        }
        ComparisonSplit::FullRight => (None, Some(full)),
    };

    if let Some(bounds) = bounds_left {
        pass.set_viewport(bounds.x, bounds.y, bounds.width, bounds.height, 0., 1.);
        passthru.draw(&mut pass, left);
    }
    if let Some(bounds) = bounds_right {
        pass.set_viewport(bounds.x, bounds.y, bounds.width, bounds.height, 0., 1.);
        passthru.draw(&mut pass, left);
    }
}
