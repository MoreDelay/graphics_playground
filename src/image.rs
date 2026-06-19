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

        #[expect(clippy::cast_precision_loss)]
        let size = iced::Size {
            width: image.width() as f32,
            height: image.height() as f32,
        };
        let draw_state = ImageDrawState {
            size,
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
            data.draw_state.viewport = viewport;
            data.render_state.draw(ctx, render_pass, &data.draw_state);
        }
    }

    pub fn zoom_in(&mut self, fix_point: Option<iced::Point>) {
        if let Some(data) = &mut self.data {
            let fix_point = match fix_point {
                Some(p) => p - data.draw_state.widget_offset(),
                None => iced::Point::default(),
            };
            data.draw_state.zoom_in(fix_point);
        }
    }

    pub fn zoom_out(&mut self, fix_point: Option<iced::Point>) {
        if let Some(data) = &mut self.data {
            let fix_point = match fix_point {
                Some(p) => p - data.draw_state.widget_offset(),
                None => iced::Point::default(),
            };
            data.draw_state.zoom_out(fix_point);
        }
    }

    pub fn set_zoom(&mut self, scale: f32, fix_point: Option<iced::Point>) {
        if let Some(data) = &mut self.data {
            let fix_point = match fix_point {
                Some(p) => p - data.draw_state.widget_offset(),
                None => iced::Point::default(),
            };
            data.draw_state.set_zoom(scale, fix_point);
        }
    }

    pub fn pan(&mut self, offset: iced::Vector) {
        if let Some(data) = &mut self.data {
            data.draw_state.pan(offset);
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
        state: &ImageDrawState,
    ) {
        let view_size = [state.viewport.width, state.viewport.height];
        let image_size = {
            let wgpu::Extent3d { width, height, .. } = self.image.size();
            #[expect(clippy::cast_precision_loss)]
            [width as f32, height as f32]
        };
        let start = [state.offset.x, state.offset.y];
        let scale = state.scale;

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
    /// Widget location and size as determined by iced layout.
    ///
    /// Needed to correct transform window to widget coordinates.
    viewport: iced::Rectangle,
    /// Image starts at this offset from the top left corner of the viewport.
    offset: iced::Point,
    /// Size of the image.
    size: iced::Size,
    /// Image is scaled by this factor.
    scale: f32,
}

impl ImageDrawState {
    const SCALE_INCREASE_FACTOR: f32 = 1.2;
    const SCALE_MAX: f32 = 15.0;
    const SCALE_MIN: f32 = 0.05;

    fn zoom_in(&mut self, fix_point: iced::Point) {
        let scale = self.scale * Self::SCALE_INCREASE_FACTOR;
        self.set_zoom(scale, fix_point);
    }

    fn zoom_out(&mut self, fix_point: iced::Point) {
        let scale = self.scale / Self::SCALE_INCREASE_FACTOR;
        self.set_zoom(scale, fix_point);
    }

    fn set_zoom(&mut self, scale: f32, fix_point: iced::Point) {
        let scale = scale.clamp(Self::SCALE_MIN, Self::SCALE_MAX);

        // get offset in fix-point coordinates (where fix-point is the origin)
        let x = self.offset.x - fix_point.x;
        let y = self.offset.y - fix_point.y;

        // scale up offset position by actual difference of scale factor
        let factor = scale / self.scale;
        let x = x * factor;
        let y = y * factor;

        // return back to viewport coordinates
        let x = x + fix_point.x;
        let y = y + fix_point.y;

        self.offset = iced::Point::new(x, y);
        self.scale = scale;
    }

    fn pan(&mut self, offset: iced::Vector) {
        self.offset += offset;
        self.clamp_offset();
    }

    /// Make sure that at least 10% of the viewport area shows part of the image.
    fn clamp_offset(&mut self) {
        const FILLED_PERCENT: f32 = 0.1;

        let x = self.offset.x.clamp(
            -self.size.width + (self.viewport.width / FILLED_PERCENT),
            self.viewport.width * (1. - FILLED_PERCENT),
        );
        let y = self.offset.y.clamp(
            -self.size.height + (self.viewport.height / FILLED_PERCENT),
            self.viewport.height * (1. - FILLED_PERCENT),
        );

        self.offset = iced::Point::new(x, y);
    }

    const fn widget_offset(&self) -> iced::Vector {
        iced::Vector {
            x: self.viewport.x,
            y: self.viewport.y,
        }
    }
}

impl Default for ImageDrawState {
    fn default() -> Self {
        Self {
            viewport: iced::Rectangle::default(),
            offset: iced::Point::default(),
            size: iced::Size::default(),
            scale: 1.,
        }
    }
}
