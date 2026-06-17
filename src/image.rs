mod gpu;

use std::path::Path;

use iced::wgpu;

use crate::{GpuContext, TargetContext};

pub struct ImageRenderState {
    image: gpu::ImageUploaded,
    _texture_layout: gpu::TextureBindGroupLayout,
    _meta_layout: gpu::ImageMetadataBindGroupLayout,
    meta_bind: gpu::ImageMetadataBinding,
    pipeline: gpu::ImagePipeline,
}

impl ImageRenderState {
    pub fn new(image: &ImageLoaded, ctx: &GpuContext, target: &TargetContext) -> Self {
        let texture_layout = gpu::TextureBindGroupLayout::new(ctx);
        let meta_layout = gpu::ImageMetadataBindGroupLayout::new(ctx);
        let uploaded = gpu::ImageUploaded::upload(image, ctx, &texture_layout);
        let meta_bind = gpu::ImageMetadataBinding::for_image(image, ctx, &meta_layout);
        let pipeline = gpu::ImagePipeline::new(ctx, target, &texture_layout, &meta_layout);
        Self {
            image: uploaded,
            _texture_layout: texture_layout,
            _meta_layout: meta_layout,
            meta_bind,
            pipeline,
        }
    }
}

impl ImageRenderState {
    pub fn draw(
        &self,
        ctx: &GpuContext,
        render_pass: &mut wgpu::RenderPass<'_>,
        viewport: iced::Rectangle,
    ) {
        let image = self.image.size();

        let raw = gpu::ImageMetadataRaw {
            view_size: [viewport.width, viewport.height],
            #[expect(clippy::cast_precision_loss)]
            image_size: [image.width as f32, image.height as f32],
            start: [0., 0.],
        };
        self.meta_bind.update(&raw, ctx);

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, self.image.bind_group(), &[]);
        render_pass.set_bind_group(1, &*self.meta_bind, &[]);
        render_pass.draw(0..4, 0..1);
    }
}

pub struct ImageLoaded {
    image: image::RgbaImage,
    format: wgpu::TextureFormat,
}

impl std::ops::Deref for ImageLoaded {
    type Target = image::RgbaImage;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

impl ImageLoaded {
    pub fn load(path: &Path) -> Result<Self, image::ImageError> {
        Self::load_as(path, wgpu::TextureFormat::Rgba8UnormSrgb)
    }

    pub fn load_as(path: &Path, format: wgpu::TextureFormat) -> Result<Self, image::ImageError> {
        let image = image::ImageReader::open(path)?
            .with_guessed_format()?
            .decode()?;
        let image = image.into();
        Ok(Self { image, format })
    }
}
