mod gpu;

use std::path::Path;

use iced::wgpu;
use iced_winit::winit::keyboard::KeyCode;

use crate::{GpuContext, TargetContext};

pub struct ImageWidget {
    data: Option<WidgetState>,
}

impl ImageWidget {
    pub const fn new() -> Self {
        let data = None;
        Self { data }
    }

    pub fn set_image(&mut self, image: ImageLoaded, ctx: &GpuContext, target: &TargetContext) {
        let features = gpu::ImagePipelineFeatures::default();
        let render_state = ImageRenderState::new(&image, ctx, target, features);

        #[expect(clippy::cast_precision_loss)]
        let size = iced::Size {
            width: image.width() as f32,
            height: image.height() as f32,
        };
        let draw_state = ImageDrawState {
            size,
            ..Default::default()
        };
        let data = WidgetState {
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
            data.draw(ctx, render_pass, viewport);
        }
    }

    pub fn zoom_in(&mut self, fix_point: Option<iced::Point>) {
        if let Some(data) = &mut self.data {
            data.zoom_in(fix_point);
        }
    }

    pub fn zoom_out(&mut self, fix_point: Option<iced::Point>) {
        if let Some(data) = &mut self.data {
            data.zoom_out(fix_point);
        }
    }

    pub fn pan(&mut self, offset: iced::Vector) {
        if let Some(data) = &mut self.data {
            data.pan(offset);
        }
    }

    pub fn update(&mut self, message: ImageMessage, cursor: Option<iced::Point>) {
        if let Some(data) = &mut self.data {
            data.update(message, cursor);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ImageMessage {
    SetZoom(f32),
    ResetPosition,
    CycleFilters,
}

impl TryFrom<KeyCode> for ImageMessage {
    type Error = ();

    fn try_from(value: KeyCode) -> Result<Self, Self::Error> {
        #[expect(clippy::wildcard_enum_match_arm)]
        match value {
            KeyCode::Digit1 => Ok(Self::SetZoom(1.)),
            KeyCode::Digit2 => Ok(Self::SetZoom(2.)),
            KeyCode::KeyS => Ok(Self::ResetPosition),
            KeyCode::KeyF => Ok(Self::CycleFilters),
            _ => Err(()),
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

struct WidgetState {
    _image: ImageLoaded,
    render_state: ImageRenderState,
    draw_state: ImageDrawState,
}

impl WidgetState {
    pub fn draw(
        &mut self,
        ctx: &GpuContext,
        render_pass: &mut wgpu::RenderPass<'_>,
        viewport: iced::Rectangle,
    ) {
        self.draw_state.viewport = viewport;
        let features = self.draw_state.pipeline_features();
        self.render_state.update_pipeline(ctx, features);
        self.render_state.draw(ctx, render_pass, &self.draw_state);
    }

    pub fn zoom_in(&mut self, cursor: Option<iced::Point>) {
        let cursor = self.draw_state.widget_pos(cursor);
        self.draw_state.zoom_in(cursor);
    }

    pub fn zoom_out(&mut self, cursor: Option<iced::Point>) {
        let cursor = self.draw_state.widget_pos(cursor);
        self.draw_state.zoom_out(cursor);
    }

    pub fn pan(&mut self, offset: iced::Vector) {
        self.draw_state.pan(offset);
    }

    pub fn update(&mut self, message: ImageMessage, cursor: Option<iced::Point>) {
        match message {
            ImageMessage::SetZoom(scale) => {
                let cursor = self.draw_state.widget_pos(cursor);
                self.draw_state.set_zoom(scale, cursor);
            }
            ImageMessage::ResetPosition => self.draw_state.reset_pos(),
            ImageMessage::CycleFilters => self.draw_state.cycle_filters(),
        }
    }
}

struct ImageRenderState {
    image: gpu::ImageUploaded,
    texture_layout: gpu::TextureBindGroupLayout,
    meta_layout: gpu::ImageMetadataBindGroupLayout,
    meta_bind: gpu::ImageMetadataBinding,
    pipeline: gpu::ImagePipeline,
}

impl ImageRenderState {
    fn new(
        image: &ImageLoaded,
        ctx: &GpuContext,
        target: &TargetContext,
        features: gpu::ImagePipelineFeatures,
    ) -> Self {
        let meta_layout = gpu::ImageMetadataBindGroupLayout::new(ctx);
        let meta_bind = gpu::ImageMetadataBinding::for_image(image, ctx, &meta_layout);

        let texture_layout = gpu::TextureBindGroupLayout::new(ctx);
        let image = gpu::ImageUploaded::upload(image, ctx, &texture_layout);

        let format = target.config.format;
        let pipeline =
            gpu::ImagePipeline::new(ctx, format, &texture_layout, &meta_layout, features);
        Self {
            image,
            texture_layout,
            meta_layout,
            meta_bind,
            pipeline,
        }
    }

    fn update_pipeline(&mut self, ctx: &GpuContext, features: gpu::ImagePipelineFeatures) {
        if self.pipeline.features() == features {
            return;
        }

        let out_format = self.pipeline.output_format();
        self.pipeline = gpu::ImagePipeline::new(
            ctx,
            out_format,
            &self.texture_layout,
            &self.meta_layout,
            features,
        );
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
    /// The image filter that should be applied
    ///
    /// Currently does not differentiate between magnification and minification.
    filter: gpu::ImageFilter,
}

impl ImageDrawState {
    const SCALE_INCREASE_FACTOR: f32 = 1.2;
    const SCALE_MAX: f32 = 15.0;
    const SCALE_MIN: f32 = 0.05;

    fn zoom_in(&mut self, fix_point: WidgetPos) {
        let scale = self.scale * Self::SCALE_INCREASE_FACTOR;
        self.set_zoom(scale, fix_point);
    }

    fn zoom_out(&mut self, fix_point: WidgetPos) {
        let scale = self.scale / Self::SCALE_INCREASE_FACTOR;
        self.set_zoom(scale, fix_point);
    }

    fn set_zoom(&mut self, scale: f32, fix_point: WidgetPos) {
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

        // when the image is at the border, it might move out of frame by zooming
        self.clamp_offset();
    }

    fn pan(&mut self, offset: iced::Vector) {
        self.offset += offset;
        self.clamp_offset();
    }

    fn reset_pos(&mut self) {
        self.offset = iced::Point::default();
        // self.clamp_offset();
    }

    fn cycle_filters(&mut self) {
        self.filter = match self.filter {
            gpu::ImageFilter::BiLinear => gpu::ImageFilter::Nearest,
            gpu::ImageFilter::Nearest => gpu::ImageFilter::BiLinear,
        };
        println!("Filter: {:?}", self.filter);
    }

    /// Make sure that at least 10% of the viewport area shows part of the image.
    fn clamp_offset(&mut self) {
        const FILLED_PERCENT: f32 = 0.1;

        let x_min = self
            .viewport
            .width
            .mul_add(FILLED_PERCENT, -self.scale * self.size.width);
        let x_max = self.viewport.width * (1. - FILLED_PERCENT);
        let x = self.offset.x.clamp(x_min, x_max);

        let y_min = self
            .viewport
            .height
            .mul_add(FILLED_PERCENT, -self.scale * self.size.height);
        let y_max = self.viewport.height * (1. - FILLED_PERCENT);
        let y = self.offset.y.clamp(y_min, y_max);

        self.offset = iced::Point::new(x, y);
    }

    fn widget_pos(&self, cursor: Option<iced::Point>) -> WidgetPos {
        let offset = iced::Vector {
            x: self.viewport.x,
            y: self.viewport.y,
        };
        let pos = cursor.map(|p| p - offset).unwrap_or_default();
        WidgetPos(pos)
    }

    fn pipeline_features(&self) -> gpu::ImagePipelineFeatures {
        let filter = self.filter;
        let mut features = gpu::ImagePipelineFeatures { filter };

        #[expect(clippy::float_cmp)]
        if self.scale == 1. {
            // When we have 100% zoom, we don't need to filter anything, as every output pixel
            // exists of exactly one texel. Disable filtering to prevent any sort of inaccuracies.
            features.filter = gpu::ImageFilter::Nearest;
        }
        features
    }
}

impl Default for ImageDrawState {
    fn default() -> Self {
        Self {
            viewport: iced::Rectangle::default(),
            offset: iced::Point::default(),
            size: iced::Size::default(),
            scale: 1.,
            filter: gpu::ImageFilter::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WidgetPos(iced::Point);

impl std::ops::Deref for WidgetPos {
    type Target = iced::Point;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
