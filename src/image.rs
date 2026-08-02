mod filters;
mod render;

use std::path::Path;

use iced::wgpu;
use iced_wgpu::core::SmolStr;
use iced_winit::winit::dpi::PhysicalSize;
use nalgebra as na;

use crate::image::filters::GaussFilter;
use crate::image::render::mipmap::MipMapper;
use crate::image::render::{ImageFilter, ImageInstruments};
use crate::instruments::bind::image::{ImageMetadataRaw, LanczosInfoRaw};
use crate::instruments::pipeline::passthru::PassThruTexture;
use crate::instruments::viewport::{VPPoint, VPVector, Viewport};
use crate::instruments::{GpuContext, TargetContext};

pub struct ImageWidget {
    image: Option<ImageLoaded>,

    instruments: ImageInstruments,

    // persistent state
    params: DrawParameters,
}

impl ImageWidget {
    const SCALE_INCREASE_FACTOR: f32 = 1.2;
    const ZOOM_MAX: f32 = 100.0;
    const ZOOM_MIN: f32 = 0.05;

    pub fn new() -> Self {
        Self {
            image: None,
            instruments: ImageInstruments::default(),
            params: DrawParameters::default(),
        }
    }

    pub fn render(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
    ) -> Option<&PassThruTexture> {
        if self.instruments.output().good().is_some() {
            return self.instruments.output().good();
        }

        let params = self.params;

        match self.params.filter {
            ImageFilter::Nearest => self.nearest(ctx, target, encoder, viewport, &params),
            ImageFilter::BiLinear => self.bilinear(ctx, target, encoder, viewport, &params),
            ImageFilter::Lanczos => self.lanczos(ctx, target, encoder, viewport, &params),
        }

        self.instruments.output().good()
    }

    pub fn update(&mut self, message: ImageMessage) {
        match message {
            ImageMessage::SetImage { image } => self.set_image(image),
            ImageMessage::ResizedViewport => self.instruments.resized(),
            ImageMessage::Pan { offset } => self.pan(offset),
            ImageMessage::SetZoom { zoom, cursor } => {
                let fixed_point = cursor.unwrap_or_else(|| VPPoint::wrap(na::Point2::origin()));
                self.set_zoom(zoom, fixed_point);
            }
            ImageMessage::ZoomIn { cursor } => {
                let fixed_point = cursor.unwrap_or_else(|| VPPoint::wrap(na::Point2::origin()));
                self.zoom_in(fixed_point);
            }
            ImageMessage::ZoomOut { cursor } => {
                let fixed_point = cursor.unwrap_or_else(|| VPPoint::wrap(na::Point2::origin()));
                self.zoom_out(fixed_point);
            }
            ImageMessage::ResetPosition => self.reset_pos(),
            ImageMessage::CycleFilters => self.cycle_filters(),
        }
    }

    fn nearest(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        params: &DrawParameters,
    ) {
        let Some(image) = self.image.as_ref() else {
            return;
        };

        render::nearest(
            image,
            &mut self.instruments,
            ctx,
            target,
            encoder,
            viewport,
            params,
        );
    }

    fn bilinear(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        params: &DrawParameters,
    ) {
        let Some(image) = self.image.as_ref() else {
            return;
        };

        render::bilinear(
            image,
            &mut self.instruments,
            ctx,
            target,
            encoder,
            viewport,
            params,
        );
    }

    fn lanczos(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        params: &DrawParameters,
    ) {
        let Some(image) = self.image.as_ref() else {
            return;
        };

        render::lanczos(
            image,
            &mut self.instruments,
            ctx,
            target,
            encoder,
            viewport,
            params,
        );
    }

    fn set_image(&mut self, image: ImageLoaded) {
        self.instruments.replaced_image();
        self.image = Some(image);
    }

    fn zoom_in(&mut self, fix_point: VPPoint) {
        let zoom = self.params.zoom * Self::SCALE_INCREASE_FACTOR;
        self.set_zoom(zoom, fix_point);
    }

    fn zoom_out(&mut self, fix_point: VPPoint) {
        let zoom = self.params.zoom / Self::SCALE_INCREASE_FACTOR;
        self.set_zoom(zoom, fix_point);
    }

    fn set_zoom(&mut self, zoom: f32, fix_point: VPPoint) {
        self.instruments.zoomed();

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

    fn pan(&mut self, offset: VPVector) {
        self.instruments.panned();
        self.params.offset += *offset;
        self.clamp_offset();
    }

    fn reset_pos(&mut self) {
        self.instruments.panned();
        self.params.offset = na::Vector2::zeros();
    }

    fn cycle_filters(&mut self) {
        self.instruments.cycled_filter();
        self.params.filter = match self.params.filter {
            ImageFilter::Nearest => ImageFilter::BiLinear,
            ImageFilter::BiLinear => ImageFilter::Lanczos,
            ImageFilter::Lanczos => ImageFilter::Nearest,
        };
        println!("Filter: {:?}", self.params.filter);
    }

    /// Make sure that at least 10% of the viewport area shows part of the image.
    fn clamp_offset(&mut self) {
        const FILLED_PERCENT: f32 = 0.1;

        let viewport = self.params.viewport;

        let size = self
            .image
            .as_ref()
            .map(ImageLoaded::size)
            .unwrap_or_default();

        #[expect(clippy::cast_precision_loss)]
        let width = viewport.width as f32;
        #[expect(clippy::cast_precision_loss)]
        let height = viewport.height as f32;

        #[expect(clippy::cast_precision_loss)]
        let x_min = width.mul_add(FILLED_PERCENT, -self.params.zoom * size.width as f32);
        let x_max = width * (1. - FILLED_PERCENT);
        let x = self.params.offset.x.clamp(x_min, x_max);

        #[expect(clippy::cast_precision_loss)]
        let y_min = height.mul_add(FILLED_PERCENT, -self.params.zoom * size.height as f32);
        let y_max = height * (1. - FILLED_PERCENT);
        let y = self.params.offset.y.clamp(y_min, y_max);

        self.params.offset = na::Vector2::new(x, y);
    }
}

#[derive(Debug, Clone)]
pub enum ImageMessage {
    SetImage { image: ImageLoaded },
    ResizedViewport,
    Pan { offset: VPVector },
    SetZoom { cursor: Option<VPPoint>, zoom: f32 },
    ZoomIn { cursor: Option<VPPoint> },
    ZoomOut { cursor: Option<VPPoint> },
    ResetPosition,
    CycleFilters,
}

impl ImageMessage {
    pub fn from_key(key: &SmolStr, cursor: Option<VPPoint>) -> Option<Self> {
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

    pub fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize {
            width: self.image.width(),
            height: self.image.height(),
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

impl std::ops::Deref for ImageLoaded {
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
