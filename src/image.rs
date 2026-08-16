pub mod filters;
mod render;

use std::path::Path;

use iced::wgpu;
use iced_wgpu::core::SmolStr;
use iced_winit::winit::dpi::PhysicalSize;
use nalgebra as na;

use crate::controls::coords::{LocalPoint, LocalVector};
use crate::image::filters::GaussFilter;
use crate::image::render::{ImageDataInstruments, ImageMetaInstruments};
use crate::instruments::bind::image::{ImageMetadataRaw, LanczosInfoRaw};
use crate::instruments::mipmap::MipMapper;
use crate::instruments::pipeline::ImageFilter;
use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};
use crate::instruments::splitview::draw_splitted;
use crate::instruments::viewport::Viewport;
use crate::instruments::{GpuContext, TargetContext};

pub struct ImageWidget {
    meta: ImageMetaInstruments,

    left: Option<SingleImageState>,
    right: Option<SingleImageState>,

    params: DrawParameters,
    split: Split,
}

impl ImageWidget {
    const SCALE_INCREASE_FACTOR: f32 = 1.2;
    const ZOOM_MAX: f32 = 100.0;
    const ZOOM_MIN: f32 = 0.05;

    pub fn new() -> Self {
        Self {
            meta: ImageMetaInstruments::new(),
            left: None,
            right: None,
            params: DrawParameters::default(),
            split: Split::default(),
        }
    }

    pub const fn split(&self) -> Option<ClampedSplit> {
        let got_two = self.left.is_some() && self.right.is_some();
        let width = self.params.viewport.width as f32;
        if got_two {
            Some(self.split.clamped(width))
        } else {
            None
        }
    }

    pub fn render(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        passthru: &PassThruPipeline,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
    ) -> Option<&PassThruTexture> {
        if self.meta.final_output().is_some() {
            return self.meta.final_output();
        }

        self.meta.uncheck_all();
        if let Some(left) = &mut self.left {
            left.instruments.uncheck_all();
        }
        if let Some(right) = &mut self.right {
            right.instruments.uncheck_all();
        }

        let params = self.params;

        match self.params.filter {
            ImageFilter::Nearest => self.nearest(ctx, target, passthru, encoder, viewport, &params),
            ImageFilter::BiLinear => {
                self.bilinear(ctx, target, passthru, encoder, viewport, &params);
            }
            ImageFilter::Lanczos => self.lanczos(ctx, target, passthru, encoder, viewport, &params),
        }

        let output = self.meta.create_output(ctx, passthru, viewport)?;

        match (&self.left, &self.right) {
            (None, _) => return None,
            (Some(left), None) => {
                let left = left.instruments.output()?;
                passthru.full_draw(encoder, left, output.view());
            }
            (Some(left), Some(right)) => {
                let left = left.instruments.output()?;
                let right = right.instruments.output()?;
                let width = params.viewport.width as f32;
                draw_splitted(
                    passthru,
                    encoder,
                    viewport,
                    left,
                    right,
                    output.view(),
                    self.split.clamped(width),
                );
            }
        }

        let output = self.meta.set_final_output(output);
        Some(output)
    }

    pub fn update(&mut self, message: ImageMessage) {
        match message {
            ImageMessage::SetImage { image } => self.set_image(image),
            ImageMessage::ResizedViewport { size } => self.resize_viewport(size),
            ImageMessage::Pan { offset } => self.pan(offset),
            ImageMessage::SetZoom { zoom, cursor } => {
                let fixed_point = cursor.unwrap_or_else(|| LocalPoint::wrap(na::Point2::origin()));
                self.set_zoom(zoom, fixed_point);
            }
            ImageMessage::ZoomIn { cursor } => {
                let fixed_point = cursor.unwrap_or_else(|| LocalPoint::wrap(na::Point2::origin()));
                self.zoom_in(fixed_point);
            }
            ImageMessage::ZoomOut { cursor } => {
                let fixed_point = cursor.unwrap_or_else(|| LocalPoint::wrap(na::Point2::origin()));
                self.zoom_out(fixed_point);
            }
            ImageMessage::ResetPosition => self.reset_pos(),
            ImageMessage::CycleFilters => self.cycle_filters(),
            ImageMessage::DragSplit { active } => self.drag_split(active),
        }
    }

    fn nearest(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        passthru: &PassThruPipeline,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        params: &DrawParameters,
    ) {
        if let Some(image) = &mut self.left {
            self.meta.nearest(
                &mut image.instruments,
                ctx,
                target,
                passthru,
                encoder,
                viewport,
                &image.data,
                params,
            );
        }

        if let Some(image) = &mut self.right {
            self.meta.nearest(
                &mut image.instruments,
                ctx,
                target,
                passthru,
                encoder,
                viewport,
                &image.data,
                params,
            );
        }
    }

    fn bilinear(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        passthru: &PassThruPipeline,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        params: &DrawParameters,
    ) {
        if let Some(image) = &mut self.left {
            self.meta.bilinear(
                &mut image.instruments,
                ctx,
                target,
                passthru,
                encoder,
                viewport,
                &image.data,
                params,
            );
        }

        if let Some(image) = &mut self.right {
            self.meta.bilinear(
                &mut image.instruments,
                ctx,
                target,
                passthru,
                encoder,
                viewport,
                &image.data,
                params,
            );
        }
    }

    fn lanczos(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        passthru: &PassThruPipeline,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        params: &DrawParameters,
    ) {
        if let Some(image) = &mut self.left {
            self.meta.lanczos(
                &mut image.instruments,
                ctx,
                target,
                passthru,
                encoder,
                viewport,
                &image.data,
                params,
            );
        }

        if let Some(image) = &mut self.right {
            self.meta.lanczos(
                &mut image.instruments,
                ctx,
                target,
                passthru,
                encoder,
                viewport,
                &image.data,
                params,
            );
        }
    }

    fn set_image(&mut self, image: ImageMemory) {
        self.meta.replaced_image();
        if let Some(left) = &mut self.left {
            left.instruments.replaced_image();
        }
        if let Some(right) = &mut self.right {
            right.instruments.replaced_image();
        }

        std::mem::swap(&mut self.left, &mut self.right);
        self.left = Some(SingleImageState::new(image));

        let mid = self.params.viewport.width / 2;
        self.split = Split::Set(ClampedSplit::Split(mid as f32));

        let default = DrawParameters::default();
        self.params = DrawParameters {
            offset: default.offset,
            zoom: default.zoom,
            ..self.params
        };
    }

    fn resize_viewport(&mut self, size: PhysicalSize<u32>) {
        self.meta.resized();
        if let Some(left) = &mut self.left {
            left.instruments.resized();
        }
        if let Some(right) = &mut self.right {
            right.instruments.resized();
        }

        self.params.viewport = size;

        if let Split::Set(split) = self.split {
            self.split = Split::Set(split.clamped(size.width as f32));
        }
    }

    fn zoom_in(&mut self, fix_point: LocalPoint) {
        let zoom = self.params.zoom * Self::SCALE_INCREASE_FACTOR;
        self.set_zoom(zoom, fix_point);
    }

    fn zoom_out(&mut self, fix_point: LocalPoint) {
        let zoom = self.params.zoom / Self::SCALE_INCREASE_FACTOR;
        self.set_zoom(zoom, fix_point);
    }

    fn set_zoom(&mut self, zoom: f32, fix_point: LocalPoint) {
        self.meta.zoomed();
        if let Some(left) = &mut self.left {
            left.instruments.zoomed();
        }
        if let Some(right) = &mut self.right {
            right.instruments.zoomed();
        }

        let zoom = zoom.clamp(Self::ZOOM_MIN, Self::ZOOM_MAX);

        // get offset in fix-point coordinates (where fix-point is the origin)
        let offset = self.params.offset - fix_point.coords;

        // scale up offset position by actual difference of scale factor
        let factor = zoom / self.params.zoom;
        let offset = offset * factor;

        // return back to viewport coordinates
        let offset = offset + fix_point.coords;

        self.params.offset = offset;
        self.params.zoom = zoom;
        println!("zoom: {zoom}");

        // when the image is at the border, it might move out of frame by zooming
        self.clamp_offset();
    }

    fn pan(&mut self, offset: LocalVector) {
        match self.split {
            Split::Set(_) => {
                self.meta.panned();
                if let Some(left) = &mut self.left {
                    left.instruments.panned();
                }
                if let Some(right) = &mut self.right {
                    right.instruments.panned();
                }

                self.params.offset += *offset;
                self.clamp_offset();
            }
            Split::Dragging(pos) => {
                // only final image is out-of-date
                self.meta.panned();

                let x = offset.x;
                self.split = Split::Dragging(pos + x);
            }
        }
    }

    fn reset_pos(&mut self) {
        self.meta.panned();
        if let Some(left) = &mut self.left {
            left.instruments.panned();
        }
        if let Some(right) = &mut self.right {
            right.instruments.panned();
        }

        self.params.offset = na::Vector2::zeros();
    }

    fn cycle_filters(&mut self) {
        self.meta.cycled_filter();
        if let Some(left) = &mut self.left {
            left.instruments.cycled_filter();
        }
        if let Some(right) = &mut self.right {
            right.instruments.cycled_filter();
        }

        self.params.filter = match self.params.filter {
            ImageFilter::Nearest => ImageFilter::BiLinear,
            ImageFilter::BiLinear => ImageFilter::Lanczos,
            ImageFilter::Lanczos => ImageFilter::Nearest,
        };
        println!("Filter: {:?}", self.params.filter);
    }

    const fn drag_split(&mut self, active: bool) {
        let width = self.params.viewport.width as f32;
        self.split = match active {
            true => self.split.dragging(width),
            false => Split::Set(self.split.clamped(width)),
        };
    }

    /// Make sure that at least 10% of the viewport area shows part of the image.
    fn clamp_offset(&mut self) {
        const FILLED_PERCENT: f32 = 0.1;

        let viewport = self.params.viewport;

        let left = self
            .left
            .as_ref()
            .map_or_else(PhysicalSize::default, |i| i.data.size());
        let right = self
            .right
            .as_ref()
            .map_or_else(PhysicalSize::default, |i| i.data.size());
        let size = left.max(right);

        let width = viewport.width as f32;
        let height = viewport.height as f32;

        let x_min = width.mul_add(FILLED_PERCENT, -self.params.zoom * size.width as f32);
        let x_max = width * (1. - FILLED_PERCENT);
        let x = self.params.offset.x.clamp(x_min, x_max);

        let y_min = height.mul_add(FILLED_PERCENT, -self.params.zoom * size.height as f32);
        let y_max = height * (1. - FILLED_PERCENT);
        let y = self.params.offset.y.clamp(y_min, y_max);

        self.params.offset = na::Vector2::new(x, y);
    }
}

#[derive(Debug, Clone, Copy)]
enum Split {
    Dragging(f32),
    Set(ClampedSplit),
}

impl Split {
    const fn clamped(self, width: f32) -> ClampedSplit {
        match self {
            Self::Dragging(pos) => ClampedSplit::new(pos, width),
            Self::Set(split) => split,
        }
    }

    const fn dragging(self, width: f32) -> Self {
        match self {
            Self::Dragging(pos) => Self::Dragging(pos),
            Self::Set(ClampedSplit::FullLeft) => Self::Dragging(0.),
            Self::Set(ClampedSplit::Split(pos)) => Self::Dragging(pos),
            Self::Set(ClampedSplit::FullRight) => Self::Dragging(width),
        }
    }
}

impl Default for Split {
    fn default() -> Self {
        Self::Set(ClampedSplit::default())
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub enum ClampedSplit {
    #[default]
    FullLeft,
    Split(f32),
    FullRight,
}

impl ClampedSplit {
    const fn new(pos: f32, width: f32) -> Self {
        if pos <= 0. {
            Self::FullLeft
        } else if pos >= width {
            Self::FullRight
        } else {
            Self::Split(pos)
        }
    }

    const fn clamped(self, width: f32) -> Self {
        match self {
            Self::Split(pos) if pos <= 0. => Self::FullLeft,
            Self::Split(pos) if pos >= width => Self::FullRight,
            keep @ (Self::Split(_) | Self::FullLeft | Self::FullRight) => keep,
        }
    }
}

struct SingleImageState {
    data: ImageMemory,
    instruments: ImageDataInstruments,
}

impl SingleImageState {
    pub fn new(data: ImageMemory) -> Self {
        let instruments = ImageDataInstruments::new();
        Self { data, instruments }
    }
}

#[derive(Clone)]
pub enum ImageMessage {
    SetImage {
        image: ImageMemory,
    },
    ResizedViewport {
        size: PhysicalSize<u32>,
    },
    Pan {
        offset: LocalVector,
    },
    SetZoom {
        cursor: Option<LocalPoint>,
        zoom: f32,
    },
    ZoomIn {
        cursor: Option<LocalPoint>,
    },
    ZoomOut {
        cursor: Option<LocalPoint>,
    },
    ResetPosition,
    CycleFilters,
    DragSplit {
        active: bool,
    },
}

impl std::fmt::Debug for ImageMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SetImage { image } => f
                .debug_struct("SetImage")
                .field(
                    "image",
                    &format!("Image({}x{})", image.width(), image.height()),
                )
                .finish(),
            Self::ResizedViewport { size } => f
                .debug_struct("ResizedViewport")
                .field("size", size)
                .finish(),
            Self::Pan { offset } => f.debug_struct("Pan").field("offset", offset).finish(),
            Self::SetZoom { cursor, zoom } => f
                .debug_struct("SetZoom")
                .field("cursor", cursor)
                .field("zoom", zoom)
                .finish(),
            Self::ZoomIn { cursor } => f.debug_struct("ZoomIn").field("cursor", cursor).finish(),
            Self::ZoomOut { cursor } => f.debug_struct("ZoomOut").field("cursor", cursor).finish(),
            Self::ResetPosition => write!(f, "ResetPosition"),
            Self::CycleFilters => write!(f, "CycleFilters"),
            Self::DragSplit { active } => {
                f.debug_struct("DragSplit").field("active", active).finish()
            }
        }
    }
}

impl ImageMessage {
    pub fn from_key(key: &SmolStr, cursor: Option<LocalPoint>) -> Option<Self> {
        match key.as_str() {
            "1" => Some(Self::SetZoom { cursor, zoom: 1. }),
            "2" => Some(Self::SetZoom { cursor, zoom: 2. }),
            "9" => Some(Self::SetZoom { cursor, zoom: 0.5 }),
            "s" => Some(Self::ResetPosition),
            "f" => Some(Self::CycleFilters),
            "-" => Some(Self::ZoomOut { cursor }),
            "+" => Some(Self::ZoomIn { cursor }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ImageMemory {
    image: image::RgbaImage,
    format: wgpu::TextureFormat,
}

impl ImageMemory {
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

    pub fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize {
            width: self.image.width(),
            height: self.image.height(),
        }
    }

    pub fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.image.width(),
            height: self.image.height(),
            depth_or_array_layers: 1,
        }
    }

    pub fn upload(&self, ctx: &GpuContext, label: Option<&str>) -> wgpu::Texture {
        let size = wgpu::Extent3d {
            width: self.image.width(),
            height: self.image.height(),
            depth_or_array_layers: 1,
        };

        let mip_level_count = size.width.min(size.height).ilog2() + 1;
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label,
            size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            // specified format above supported by default, only additional view formats here
            view_formats: &[],
        });

        // assuming only uncompressed formats are used here
        let texel_bytes = self
            .format
            .block_copy_size(None)
            .expect("assuming no complex texture format is used");

        // load image (on CPU) into texture (on GPU) by issuing command over queue
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &self.image,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(texel_bytes * size.width),
                rows_per_image: Some(size.height),
            },
            size,
        );

        let mipmapper = MipMapper::new(ctx);
        mipmapper.compute_mipmaps(ctx, &texture);

        texture
    }
}

impl std::ops::Deref for ImageMemory {
    type Target = image::RgbaImage;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

#[derive(Debug, Copy, Clone, PartialEq)]
struct DrawParameters {
    /// Widget size as determined by iced layout.
    viewport: PhysicalSize<u32>,
    /// Image starts at this offset from the top left corner of the viewport.
    offset: na::Vector2<f32>,
    /// Image is scaled by this factor.
    zoom: f32,
    /// The image filter that should be applied
    ///
    /// Currently does not differentiate between magnification and minification.
    filter: ImageFilter,
}

impl DrawParameters {
    fn raw_metadata(&self) -> ImageMetadataRaw {
        ImageMetadataRaw {
            start: [self.offset.x, self.offset.y],
            zoom: self.zoom,
            _pad: 0,
        }
    }

    #[expect(clippy::unused_self)]
    const fn raw_lanczos(&self) -> LanczosInfoRaw {
        LanczosInfoRaw { filter_size: 2. }
    }

    fn raw_blur_kernel(&self) -> Option<Vec<f32>> {
        // This factor is a trade-off between sharpness (lower) and anti-aliasing (higher). 0.3
        // looks the best from testing around.
        const FACTOR: f32 = 0.3;

        if self.zoom >= 1. {
            return None;
        }

        let sigma = FACTOR / self.zoom;
        let kernel = GaussFilter::new(sigma)?;
        Some(kernel.blur_kernel())
    }
}

impl Default for DrawParameters {
    fn default() -> Self {
        Self {
            viewport: PhysicalSize::default(),
            offset: na::Matrix::default(),
            zoom: 1.,
            filter: ImageFilter::default(),
        }
    }
}
