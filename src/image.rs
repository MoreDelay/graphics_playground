mod gpu;

use std::path::Path;

use iced::wgpu;

use crate::{GpuContext, TargetContext};

pub struct ImageWidget {
    data: Option<WidgetData>,
}

impl ImageWidget {
    pub const fn new() -> Self {
        let data = None;
        Self { data }
    }

    pub fn set_image(&mut self, image: ImageLoaded, ctx: &GpuContext, target: &TargetContext) {
        let render_state = ImageRenderState::new(&image, ctx, target);
        let draw_state = ImageDrawState::default();
        let data = WidgetData {
            image,
            render_state,
            draw_state,
        };
        self.data = Some(data);
    }

    pub fn draw(
        &mut self,
        ctx: &GpuContext,
        render_pass: &mut wgpu::RenderPass<'_>,
        viewport: iced::Rectangle,
    ) {
        if let Some(data) = &mut self.data {
            let iced::Rectangle {
                x,
                y,
                width,
                height,
            } = viewport;
            #[expect(clippy::cast_possible_truncation)]
            let size = iced::Size {
                width: (width - x.fract()).ceil() as u32,
                height: (height - y.fract()).ceil() as u32,
            };

            data.draw_state.viewport = size;
            data.render_state
                .draw(ctx, render_pass, viewport, &data.draw_state);
        }
    }

    pub fn zoom_in(&mut self) {
        if let Some(data) = &mut self.data {
            data.draw_state.zoom_in();
        }
    }

    pub fn zoom_out(&mut self) {
        if let Some(data) = &mut self.data {
            data.draw_state.zoom_out();
        }
    }

    pub fn pan(&mut self, x: i32, y: i32) {
        if let Some(data) = &mut self.data {
            data.draw_state.pan(x, y, &data.image);
        }
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

    pub fn width(&self) -> u32 {
        self.image.width()
    }

    pub fn height(&self) -> u32 {
        self.image.height()
    }
}

struct WidgetData {
    image: ImageLoaded,
    render_state: ImageRenderState,
    draw_state: ImageDrawState,
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
        options: &ImageDrawState,
    ) {
        let view_size = [viewport.width, viewport.height];
        let image_size = {
            let wgpu::Extent3d { width, height, .. } = self.image.size();
            #[expect(clippy::cast_precision_loss)]
            [width as f32, height as f32]
        };
        #[expect(clippy::cast_precision_loss)]
        let start = [options.offset.x as f32, options.offset.y as f32];
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

struct ImageDrawState {
    /// Image starts at this offset from the top left corner of the viewport.
    offset: iced::Point<i32>,
    /// Image is scaled by this factor.
    scale: f32,
    /// Size of the viewport where the image is shown (from the last draw)
    viewport: iced::Size<u32>,
}

impl ImageDrawState {
    const SCALE_INCREASE_FACTOR: f32 = 1.1;

    pub fn zoom_in(&mut self) {
        let scale = self.scale * Self::SCALE_INCREASE_FACTOR;
        let scale = scale.min(15.);
        self.scale = scale;
    }

    pub fn zoom_out(&mut self) {
        let scale = self.scale / Self::SCALE_INCREASE_FACTOR;
        let scale = scale.max(0.05);
        self.scale = scale;
    }

    pub fn pan(&mut self, x: i32, y: i32, image: &ImageLoaded) {
        let x = self.offset.x + x;
        let y = self.offset.y + y;
        #[expect(clippy::cast_possible_wrap)]
        let x = x.clamp(
            -(image.width() as i32) + (self.viewport.width / 10) as i32,
            (self.viewport.width * 9 / 10) as i32,
        );
        #[expect(clippy::cast_possible_wrap)]
        let y = y.clamp(
            -(image.height() as i32) + (self.viewport.height / 10) as i32,
            (self.viewport.height * 9 / 10) as i32,
        );
        let offset = iced::Point::new(x, y);
        self.offset = offset;
    }
}

impl Default for ImageDrawState {
    fn default() -> Self {
        Self {
            offset: iced::Point::new(0, 0),
            scale: 1.,
            viewport: iced::Size {
                width: 0,
                height: 0,
            },
        }
    }
}
