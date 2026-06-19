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
        let draw_state = ImageDrawState {
            image: image.size(),
            ..Default::default()
        };
        let data = WidgetData {
            _image: image,
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
            data.draw_state.pan(x, y);
        }
    }

    pub fn set_zoom(&mut self, scale: f32) {
        if let Some(data) = &mut self.data {
            data.draw_state.set_zoom(scale);
        }
    }
}

pub struct ImageLoaded {
    image: image::RgbaImage,
    format: wgpu::TextureFormat,
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

    pub fn size(&self) -> iced::Size<u32> {
        iced::Size {
            width: self.image.width(),
            height: self.image.height(),
        }
    }
}

impl std::ops::Deref for ImageLoaded {
    type Target = image::RgbaImage;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

struct WidgetData {
    _image: ImageLoaded,
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
    /// Size of the viewport where the image is shown (from the last draw).
    viewport: iced::Size<u32>,
    /// Size of the image.
    image: iced::Size<u32>,
}

impl ImageDrawState {
    const SCALE_INCREASE_FACTOR: f32 = 1.1;
    const SCALE_MAX: f32 = 15.0;
    const SCALE_MIN: f32 = 0.05;

    fn zoom_in(&mut self) {
        self.scale *= Self::SCALE_INCREASE_FACTOR;
        self.clamp_values();
    }

    fn zoom_out(&mut self) {
        self.scale /= Self::SCALE_INCREASE_FACTOR;
        self.clamp_values();
    }

    fn pan(&mut self, x: i32, y: i32) {
        self.offset.x += x;
        self.offset.y += y;
        self.clamp_values();
    }

    fn set_zoom(&mut self, scale: f32) {
        self.scale = scale;
        self.clamp_values();
    }

    #[expect(clippy::cast_precision_loss)]
    fn clamp_values(&mut self) {
        self.scale = self.scale.clamp(Self::SCALE_MIN, Self::SCALE_MAX);

        let x = self.offset.x as f32;
        let y = self.offset.y as f32;
        let image_width = self.image.width as f32 * self.scale;
        let image_height = self.image.height as f32 * self.scale;
        let view_width = self.viewport.width as f32;
        let view_height = self.viewport.height as f32;

        let x = x.clamp(-image_width + (view_width / 10.), view_width * 9. / 10.);
        let y = y.clamp(-image_height + (view_height / 10.), view_height * 9. / 10.);

        #[expect(clippy::cast_possible_truncation)]
        let offset = iced::Point::new(x as i32, y as i32);
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
            image: iced::Size {
                width: 0,
                height: 0,
            },
        }
    }
}
