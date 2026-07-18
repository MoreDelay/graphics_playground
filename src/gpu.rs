pub mod bind;
mod buffer;
pub mod pipeline;

use std::path::Path;
use std::sync::Arc;

use iced::wgpu;
use iced_winit::winit;
use image::{ImageBuffer, Rgba};
use wesl::Wesl;
use winit::window::Window;

#[rustfmt::skip]
#[expect(unused)]
pub use buffer::BufferRaw;
pub use buffer::SimpleBuffer;

pub const SHADER_ROOT: &str = "src/shader";

pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

pub struct TargetContext {
    pub window: Arc<Window>,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
}

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

#[expect(unused, reason = "currently only ever used for debugging")]
pub fn store_texture_as_image(ctx: &GpuContext, texture: &wgpu::Texture, image_path: &Path) {
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

    let size = width * height * 4;
    let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Debug Image Temporary Buffer"),
        size: size as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

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
                bytes_per_row: Some(4 * width),
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
        let image = ImageBuffer::<Rgba<u8>, _>::from_raw(width, height, data)
            .expect("texture should fit into specified dimensions, checked before");
        image.save(image_path).expect("saving to disk should work");
    }
    buffer.unmap();
}
