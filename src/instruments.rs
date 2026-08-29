//! Contains all rendering instruments
//!
//! Instruments are wrappers around wgpu objects designed to perform or be used for a specific
//! purpose. This makes it easier to orchestrate more complex behavior without accidentally using
//! the wrong handles.
//!
//! For example, to use a resource in a shader, we need to use some [`wgpu::BindGroup`]. However,
//! behind this handle can be many different kinds of resources, such as buffers (of different
//! data), textures, etc. To not mix these up, we wrap all these handles in concrete types to easily
//! differentiate between them, such as [`buffer::SimpleBuffer`] or
//! [`bind::texture::SimpleTexture`].

pub mod bind;
pub mod buffer;
pub mod mesh;
pub mod mipmap;
pub mod pipeline;
pub mod splitview;
pub mod viewport;

use std::path::Path;
use std::sync::Arc;

use iced::wgpu;
use iced_winit::winit;
use image::{ImageBuffer, Rgba};
use wesl::Wesl;
use winit::window::Window;

/// Directory root where shaders are located
pub const SHADER_ROOT: &str = "src/shader";

/// Context to work with the gpu
pub struct GpuContext {
    /// The global [`wgpu::Device`]
    pub device: wgpu::Device,
    /// The global [`wgpu::Queue`]
    pub queue: wgpu::Queue,
}

/// Context to work with the window
pub struct TargetContext {
    /// The window that we draw to
    pub window: Arc<Window>,
    /// The surface configures for wgpu
    pub surface: wgpu::Surface<'static>,
    /// The current configuration of our surface
    pub config: wgpu::SurfaceConfiguration,
}

/// Helper to create a simple shader module description with no features
pub fn create_simple_shader_module_desc<'a>(
    label: Option<&'a str>,
    wesl_path: &str,
) -> wgpu::ShaderModuleDescriptor<'a> {
    let compute_module = wesl_path.parse().expect("module path invalid");
    let compute_module = Wesl::new(SHADER_ROOT)
        .compile(&compute_module)
        .inspect_err(|e| eprintln!("WESL error: {e}"))
        .expect("shader invalid")
        .to_string();
    wgpu::ShaderModuleDescriptor {
        label,
        source: wgpu::ShaderSource::Wgsl(compute_module.into()),
    }
}

/// Debug helper to download and save a texture as an image
#[expect(dead_code, reason = "currently only ever used for debugging")]
pub fn save_texture_as_image(ctx: &GpuContext, texture: &wgpu::Texture, image_path: &Path) {
    assert!(
        matches!(
            texture.format(),
            wgpu::TextureFormat::Rgba8Unorm | wgpu::TextureFormat::Rgba8UnormSrgb
        ),
        "only rgba supported atm"
    );

    let mip_level = 0;
    let width = texture.width();
    let height = texture.height();
    let bytes_per_row = 4 * width;

    let padded_bytes_per_row = bytes_per_row.next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
    let padded_size = height * padded_bytes_per_row;

    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Debug Image Temporary Buffer"),
        size: padded_size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Debug Image Encoder"),
        });

    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(height),
            },
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
    );

    ctx.queue.submit(Some(encoder.finish()));

    {
        let slice = buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |res| res.expect("copy should succeed"));
        ctx.device
            .poll(wgpu::PollType::wait_indefinitely())
            .expect("single threaded wait should succeed");

        let data = slice.get_mapped_range();
        let image = data
            .chunks_exact(padded_bytes_per_row as usize)
            .flat_map(|c| &c[..bytes_per_row as usize])
            .copied()
            .collect::<Vec<_>>();
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, image)
            .expect("texture should fit into specified dimensions, checked before");
        image.save(image_path).expect("saving to disk should work");
    }
    buffer.unmap();
}
