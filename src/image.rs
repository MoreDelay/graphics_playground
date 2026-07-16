mod gpu;

use std::path::Path;

use iced::wgpu;
use iced_wgpu::core::SmolStr;
use iced_winit::winit::dpi::{LogicalInsets, LogicalPosition, PhysicalInsets};

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
        viewport: LogicalInsets<f32>,
        scale_factor: f64,
    ) {
        if let Some(data) = &mut self.data {
            data.draw(ctx, render_pass, viewport, scale_factor);
        }
    }

    pub fn pan(&mut self, offset: iced::Vector) {
        if let Some(data) = &mut self.data {
            data.pan(offset);
        }
    }

    pub fn update(&mut self, message: ImageMessage) {
        if let Some(data) = &mut self.data {
            data.update(message);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ImageMessage {
    SetZoom {
        cursor: Option<iced::Point>,
        scale: f32,
    },
    ZoomIn {
        cursor: Option<iced::Point>,
    },
    ZoomOut {
        cursor: Option<iced::Point>,
    },
    ResetPosition,
    CycleFilters,
}

impl ImageMessage {
    pub fn from_key(key: &SmolStr, cursor: Option<iced::Point>) -> Option<Self> {
        match key.as_str() {
            "1" => Some(Self::SetZoom { cursor, scale: 1. }),
            "2" => Some(Self::SetZoom { cursor, scale: 2. }),
            "9" => Some(Self::SetZoom { cursor, scale: 0.5 }),
            "s" => Some(Self::ResetPosition),
            "f" => Some(Self::CycleFilters),
            "-" => Some(Self::ZoomOut { cursor }),
            "+" => Some(Self::ZoomIn { cursor }),
            _ => None,
        }
    }
}

pub struct ImageLoaded {
    image: image::RgbaImage,
    format: wgpu::TextureFormat,
}

impl ImageLoaded {
    pub const FORMAT_SRGB: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    pub fn load(path: &Path) -> Result<Self, image::ImageError> {
        Self::load_as(path, Self::FORMAT_SRGB)
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
        viewport: LogicalInsets<f32>,
        scale_factor: f64,
    ) {
        self.draw_state.viewport = viewport;
        let features = self.draw_state.pipeline_features();
        self.render_state.update_pipeline(ctx, features);
        self.render_state
            .draw(ctx, render_pass, &self.draw_state, scale_factor);
    }

    pub fn pan(&mut self, offset: iced::Vector) {
        self.draw_state.pan(offset);
    }

    pub fn update(&mut self, message: ImageMessage) {
        match message {
            ImageMessage::SetZoom { scale, cursor } => {
                let cursor = self.draw_state.widget_pos(cursor);
                self.draw_state.set_zoom(scale, cursor);
            }
            ImageMessage::ZoomIn { cursor } => {
                let cursor = self.draw_state.widget_pos(cursor);
                self.draw_state.zoom_in(cursor);
            }
            ImageMessage::ZoomOut { cursor } => {
                let cursor = self.draw_state.widget_pos(cursor);
                self.draw_state.zoom_out(cursor);
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
        scale_factor: f64,
    ) {
        self.meta_bind
            .update(ctx, &state.as_image_metadata(scale_factor));

        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, self.image.bind_group(), &[]);
        render_pass.set_bind_group(1, &*self.meta_bind, &[]);
        render_pass.draw(0..4, 0..1);
    }
}

struct ImageDrawState {
    /// Widget location and size as determined by iced layout.
    ///
    /// Needed to transform window to widget coordinates.
    viewport: LogicalInsets<f32>,
    /// Image starts at this offset from the top left corner of the viewport.
    offset: LogicalPosition<f32>,
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
    const SCALE_MAX: f32 = 100.0;
    const SCALE_MIN: f32 = 0.05;

    fn as_image_metadata(&self, scale_factor: f64) -> gpu::ImageMetadataRaw {
        let viewport: PhysicalInsets<f32> = self.viewport.to_physical(scale_factor);
        let width = viewport.right - viewport.left;
        let height = viewport.bottom - viewport.top;

        let offset = self.offset.to_physical(scale_factor);

        gpu::ImageMetadataRaw {
            view_size: [width, height],
            start: [offset.x, offset.y],
            scale: self.scale,
            _pad: 0,
        }
    }

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

        self.offset = LogicalPosition::new(x, y);
        self.scale = scale;
        println!("scale: {scale}");

        // when the image is at the border, it might move out of frame by zooming
        self.clamp_offset();
    }

    fn pan(&mut self, offset: iced::Vector) {
        self.offset.x += offset.x;
        self.offset.y += offset.y;
        self.clamp_offset();
    }

    fn reset_pos(&mut self) {
        self.offset = LogicalPosition::default();
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

        let width = self.viewport.right - self.viewport.left;
        let height = self.viewport.bottom - self.viewport.top;
        let x_min = width.mul_add(FILLED_PERCENT, -self.scale * self.size.width);
        let x_max = width * (1. - FILLED_PERCENT);
        let x = self.offset.x.clamp(x_min, x_max);

        let y_min = height.mul_add(FILLED_PERCENT, -self.scale * self.size.height);
        let y_max = height * (1. - FILLED_PERCENT);
        let y = self.offset.y.clamp(y_min, y_max);

        self.offset = LogicalPosition::new(x, y);
    }

    fn widget_pos(&self, cursor: Option<iced::Point>) -> WidgetPos {
        let Some(cursor) = cursor else {
            return WidgetPos(LogicalPosition::default());
        };
        let offset = iced::Vector {
            x: self.viewport.left,
            y: self.viewport.top,
        };
        let iced::Point { x, y } = cursor - offset;
        WidgetPos(LogicalPosition::new(x, y))
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
            viewport: LogicalInsets::default(),
            offset: LogicalPosition::default(),
            size: iced::Size::default(),
            scale: 1.,
            filter: gpu::ImageFilter::default(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct WidgetPos(LogicalPosition<f32>);

impl std::ops::Deref for WidgetPos {
    type Target = LogicalPosition<f32>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn inset_to_rectangle(inset: LogicalInsets<f32>, scale_factor: f64) -> iced::Rectangle {
    let PhysicalInsets {
        top,
        left,
        bottom,
        right,
    } = inset.to_physical(scale_factor);

    iced::Rectangle {
        x: left,
        y: top,
        width: right - left,
        height: bottom - top,
    }
}
