mod gpu;

use std::path::Path;

use iced::wgpu;

use crate::image::gpu::{
    ImageMetadataBindGroupLayout,
    ImageMetadataBinding,
    ImagePipeline,
    ImageUploaded,
    TextureBindGroupLayout,
};
use crate::{GpuContext, TargetContext};

pub struct ImageRenderState {
    image: ImageUploaded,
    _texture_layout: TextureBindGroupLayout,
    _meta_layout: ImageMetadataBindGroupLayout,
    meta_bind: ImageMetadataBinding,
    pipeline: ImagePipeline,
}

impl ImageRenderState {
    pub fn new(image: &ImageLoaded, ctx: &GpuContext, target: &TargetContext) -> Self {
        let texture_layout = TextureBindGroupLayout::new(ctx);
        let meta_layout = ImageMetadataBindGroupLayout::new(ctx);
        let uploaded = image.upload(ctx, &texture_layout);
        let meta_bind = ImageMetadataBinding::for_image(image, ctx, &meta_layout);
        let pipeline = ImagePipeline::new(ctx, target, &texture_layout, &meta_layout);
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
    pub fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
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

    fn upload(&self, ctx: &GpuContext, layout: &TextureBindGroupLayout) -> ImageUploaded {
        ImageUploaded::upload(self, ctx, layout)
    }
}
