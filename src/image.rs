mod gpu;

use std::path::Path;

use iced::wgpu;

use crate::{GpuContext, TargetContext};

pub struct ImageWidget {
    data: Option<WidgetData>,
    draw_state: ImageDrawOptions,
}

impl ImageWidget {
    const SCALE_INCREASE_FACTOR: f32 = 1.1;

    pub fn new() -> Self {
        let data = None;
        let draw_state = ImageDrawOptions::default();
        Self { data, draw_state }
    }

    pub fn set_image(&mut self, image: ImageLoaded, ctx: &GpuContext, target: &TargetContext) {
        let render_state = ImageRenderState::new(&image, ctx, target);
        let data = WidgetData {
            _image: image,
            render_state,
        };
        self.data = Some(data);
    }

    pub fn draw(
        &self,
        ctx: &GpuContext,
        render_pass: &mut wgpu::RenderPass<'_>,
        viewport: iced::Rectangle,
    ) {
        if let Some(data) = &self.data {
            data.render_state
                .draw(ctx, render_pass, viewport, &self.draw_state);
        }
    }

    pub fn zoom_in(&mut self) -> f32 {
        let scale = self.draw_state.scale * Self::SCALE_INCREASE_FACTOR;
        let scale = scale.min(15.);
        self.draw_state.scale = scale;
        scale
    }

    pub fn zoom_out(&mut self) -> f32 {
        let scale = self.draw_state.scale / Self::SCALE_INCREASE_FACTOR;
        let scale = scale.max(0.05);
        self.draw_state.scale = scale;
        scale
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

struct WidgetData {
    _image: ImageLoaded,
    render_state: ImageRenderState,
}

struct ImageRenderState {
    image: gpu::ImageUploaded,
    _texture_layout: gpu::TextureBindGroupLayout,
    _meta_layout: gpu::ImageMetadataBindGroupLayout,
    meta_bind: gpu::ImageMetadataBinding,
    pipeline: gpu::ImagePipeline,
}

impl ImageRenderState {
    fn new(image: &ImageLoaded, ctx: &GpuContext, target: &TargetContext) -> Self {
        let meta_layout = gpu::ImageMetadataBindGroupLayout::new(ctx);
        let meta_bind = gpu::ImageMetadataBinding::for_image(image, ctx, &meta_layout);

        let texture_layout = gpu::TextureBindGroupLayout::new(ctx);
        let image = gpu::ImageUploaded::upload(image, ctx, &texture_layout);

        let pipeline = gpu::ImagePipeline::new(ctx, target, &texture_layout, &meta_layout);
        Self {
            image,
            _texture_layout: texture_layout,
            _meta_layout: meta_layout,
            meta_bind,
            pipeline,
        }
    }

    fn draw(
        &self,
        ctx: &GpuContext,
        render_pass: &mut wgpu::RenderPass<'_>,
        viewport: iced::Rectangle,
        options: &ImageDrawOptions,
    ) {
        let view_size = [viewport.width, viewport.height];
        let image_size = {
            let wgpu::Extent3d { width, height, .. } = self.image.size();
            #[expect(clippy::cast_precision_loss)]
            [width as f32, height as f32]
        };
        #[expect(clippy::cast_precision_loss)]
        let start = [options.offset.0 as f32, options.offset.1 as f32];
        let scale = options.scale;

        let raw = gpu::ImageMetadataRaw {
            view_size,
            image_size,
            start,
            scale,
            _pad: 0,
        };
        self.meta_bind.update(ctx, &raw);

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, self.image.bind_group(), &[]);
        render_pass.set_bind_group(1, &*self.meta_bind, &[]);
        render_pass.draw(0..4, 0..1);
    }
}

struct ImageDrawOptions {
    /// Image starts at this offset from the top left corner of the viewport.
    offset: (i32, i32),
    /// Image is scaled by this factor.
    scale: f32,
}

impl Default for ImageDrawOptions {
    fn default() -> Self {
        Self {
            offset: (0, 0),
            scale: 1.,
        }
    }
}
