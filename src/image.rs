mod filters;
mod render;

use std::path::Path;
use std::range::Range;

use iced::wgpu;
use iced_wgpu::core::SmolStr;
use iced_winit::winit::dpi::PhysicalSize;
use nalgebra as na;

use crate::image::filters::GaussFilter;
use crate::image::render::mipmap::MipMapper;
use crate::instruments::bind::image::{ImageMetadataRaw, LanczosInfoRaw};
use crate::instruments::bind::storage::{SimpleStorageTexture, StorageTextureCopyMachine};
use crate::instruments::bind::texture::{SimpleTexture, SimpleTextureLayout};
use crate::instruments::buffer::{SimpleBuffer, SimpleBufferBind, SimpleBufferBindLayout};
use crate::instruments::pipeline::filter::{
    ConvolutionPipeline, ConvolutionPipelineLayout, KernelBinding, KernelLayout,
    StorageSrcDstLayout,
};
use crate::instruments::pipeline::image::{
    LanczosImageRenderPipelineLayout, RenderBilinearPipeline, RenderLanczosPipeline,
    RenderNearestPipeline, SimpleImageRenderPipelineLayout,
};
use crate::instruments::pipeline::passthru::PassThruTexture;
use crate::instruments::viewport::{VPPoint, VPVector, Viewport};
use crate::instruments::{GpuContext, TargetContext};

pub struct ImageWidget {
    data: WidgetState,
}

impl ImageWidget {
    pub fn new() -> Self {
        let data = WidgetState::new();
        Self { data }
    }

    pub fn current_render_output(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
    ) -> Option<&PassThruTexture> {
        self.data
            .current_render_output(ctx, target, encoder, viewport)
    }

    pub fn update(&mut self, message: ImageMessage) {
        self.data.update(message);
    }
}

#[derive(Debug, Clone)]
pub enum ImageMessage {
    SetImage { image: ImageLoaded },
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

struct WidgetState {
    image: Option<ImageLoaded>,

    instruments: ImageInstruments,

    // persistent state
    params: DrawParameters,
}

impl WidgetState {
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
        output: &PassThruTexture,
        params: DrawParameters,
    ) {
        let zoom = params.zoom;

        todo!()
    }

    pub fn current_render_output(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
    ) -> Option<&PassThruTexture> {
        if self.instruments.output.get().is_some() {
            return self.instruments.output.get();
        }

        let params = self.params.clone();

        self.lanczos(ctx, target, encoder, viewport, &params);
        // self.nearest(ctx, target, encoder, viewport, &params);

        let res = self.instruments.output.get();
        dbg!(res.is_some());
        res
    }

    pub fn update(&mut self, message: ImageMessage) {
        match message {
            ImageMessage::SetImage { image } => self.set_image(image),
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
        println!("run nearest");

        let Some(image) = self.image.as_ref() else {
            return;
        };

        let output = self
            .instruments
            .output
            .or_set_outdated(|| viewport.create_texture(ctx).expect("should work"));
        if let InstrumentAvailable::Valid(_) = output {
            return;
        }

        let texture_layout = self
            .instruments
            .texture_layout
            .or_replace(|| SimpleTextureLayout::new(ctx, None));
        let original = self.instruments.original.or_replace(|| {
            let texture = image.upload(ctx, None);
            SimpleTexture::new(ctx, texture_layout, texture, None)
        });

        let buffer_layout = self
            .instruments
            .buffer_layout
            .or_replace(|| SimpleBufferBindLayout::new(ctx, None));
        let pipeline_layout = self.instruments.simple_pipeline_layout.or_replace(|| {
            SimpleImageRenderPipelineLayout::new(ctx, texture_layout, buffer_layout)
        });
        let pipeline = self
            .instruments
            .nearest_pipeline
            .or_replace(|| RenderNearestPipeline::new(ctx, pipeline_layout, target.config.format));

        let meta = params.raw_metadata();
        let meta_buffer = self
            .instruments
            .meta_buffer
            .or_set(|| {
                let meta_buffer = SimpleBuffer::new(ctx, meta, None);
                SimpleBufferBind::new(ctx, meta_buffer, buffer_layout, None)
            })
            .or_update(|b| b.buffer().update(ctx, meta));

        output.or_update(|output| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lanczos Image Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output.view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pipeline.draw(&mut pass, original, meta_buffer);
        });

        println!("drawing everything");
    }

    fn lanczos(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        params: &DrawParameters,
    ) {
        println!("run lanczos");

        let Some(image) = self.image.as_ref() else {
            return;
        };

        let output = self
            .instruments
            .output
            .or_set_outdated(|| viewport.create_texture(ctx).expect("should work"));
        if let InstrumentAvailable::Valid(_) = output {
            return;
        }

        let texture_layout = self
            .instruments
            .texture_layout
            .or_replace(|| SimpleTextureLayout::new(ctx, None));
        let original = self.instruments.original.or_replace(|| {
            let texture = image.upload(ctx, None);
            SimpleTexture::new(ctx, texture_layout, texture, None)
        });
        let original_texture = original.texture();

        let (storage_layout, kernel_layout, convolution_pipeline) = {
            let storage = self
                .instruments
                .storage_layout
                .or_replace(|| StorageSrcDstLayout::new(ctx, None));
            let kernel = self
                .instruments
                .kernel_layout
                .or_replace(|| KernelLayout::new(ctx, None));
            let convolution = self
                .instruments
                .convolution_layout
                .or_replace(|| ConvolutionPipelineLayout::new(ctx, storage, kernel, None));
            let pipeline = self
                .instruments
                .convolution_pipeline
                .or_replace(|| ConvolutionPipeline::new(ctx, convolution, None));
            (storage, kernel, pipeline)
        };

        let copy_machine = self
            .instruments
            .copy_machine
            .or_replace(|| StorageTextureCopyMachine::new(ctx, original_texture.format()));

        let (storage_data, storage_scratch) = {
            let storage_data = self
                .instruments
                .storage_data
                .or_replace(|| SimpleStorageTexture::empty(ctx, original_texture, None));
            let storage_scratch = self
                .instruments
                .storage_scratch
                .or_replace(|| SimpleStorageTexture::empty(ctx, original_texture, None));
            (storage_data, storage_scratch)
        };
        storage_data.copy_from_texture(
            ctx,
            encoder,
            copy_machine,
            original_texture,
            Range::from(0..1),
        );

        let kernel_bind = self.instruments.kernel_bind.or_replace(|| {
            let kernel = params.raw_blur_kernel();
            KernelBinding::new(ctx, kernel_layout, &kernel, None)
        });
        convolution_pipeline.run(
            ctx,
            encoder,
            storage_layout,
            storage_data,
            storage_scratch,
            kernel_bind,
            0,
        );

        let blurred = self
            .instruments
            .blurred
            .or_replace(|| SimpleTexture::empty(ctx, texture_layout, original_texture, None));
        storage_data.copy_to_texture(
            ctx,
            encoder,
            copy_machine,
            blurred.texture(),
            Range::from(0..1),
        );

        let buffer_layout = self
            .instruments
            .buffer_layout
            .or_replace(|| SimpleBufferBindLayout::new(ctx, None));
        let pipeline_layout = self.instruments.lanczos_pipeline_layout.or_replace(|| {
            LanczosImageRenderPipelineLayout::new(ctx, texture_layout, buffer_layout)
        });
        let pipeline = self
            .instruments
            .lanczos_pipeline
            .or_replace(|| RenderLanczosPipeline::new(ctx, pipeline_layout, target.config.format));

        let meta = params.raw_metadata();
        let meta_buffer = self
            .instruments
            .meta_buffer
            .or_set(|| {
                let meta_buffer = SimpleBuffer::new(ctx, meta, None);
                SimpleBufferBind::new(ctx, meta_buffer, buffer_layout, None)
            })
            .or_update(|b| b.buffer().update(ctx, meta));
        let lanczos = params.raw_lanczos();
        let lanczos_buffer = self
            .instruments
            .lanczos_buffer
            .or_set(|| {
                let lanczos = SimpleBuffer::new(ctx, lanczos, None);
                SimpleBufferBind::new(ctx, lanczos, buffer_layout, None)
            })
            .or_update(|b| b.buffer().update(ctx, lanczos));

        output.or_update(|output| {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lanczos Image Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: output.view(),
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            pipeline.draw(&mut pass, blurred, meta_buffer, lanczos_buffer);
        });

        println!("drawing everything");
    }

    fn set_image(&mut self, image: ImageLoaded) {
        println!("update image");
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

struct LanczosState {
    blurred: wgpu::Texture,
    original_bind: SimpleTexture,
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

    fn raw_lanczos(&self) -> LanczosInfoRaw {
        LanczosInfoRaw { filter_size: 2. }
    }

    fn raw_blur_kernel(&self) -> Vec<f32> {
        let sigma = self.zoom; // TODO: how to derive sigma from zoom?
        let kernel = GaussFilter::new(sigma).expect("valid sigma");
        kernel.blur_kernel()
    }
}

impl Default for DrawParameters {
    fn default() -> Self {
        Self {
            viewport: Default::default(),
            offset: Default::default(),
            zoom: 1.,
            filter: Default::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFilter {
    #[default]
    Nearest,
    BiLinear,
    Lanczos,
}

#[derive(Default)]
enum Instrument<T> {
    #[default]
    Missing,
    Available(InstrumentAvailable<T>),
}

impl<T> Instrument<T> {
    fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    fn unwrap_outdated(self) -> T {
        match self {
            Self::Missing => panic!("value is missing"),
            Self::Available(t) => t.unwrap_outdated(),
        }
    }

    fn unwrap(self) -> T {
        match self {
            Self::Missing => panic!("value is missing"),
            Self::Available(t) => t.unwrap(),
        }
    }

    fn unwrap_any(self) -> T {
        match self {
            Self::Missing => panic!("value is missing"),
            Self::Available(t) => t.unwrap_any(),
        }
    }

    fn or_set<F>(&mut self, f: F) -> &mut InstrumentAvailable<T>
    where
        F: FnOnce() -> T,
    {
        let f = || InstrumentAvailable::Valid(f());
        self.or_use(f)
    }

    fn or_set_outdated<F>(&mut self, f: F) -> &mut InstrumentAvailable<T>
    where
        F: FnOnce() -> T,
    {
        match self {
            Self::Missing => {
                let inner = InstrumentAvailable::OutOfDate(f());
                *self = Self::Available(inner);
            }
            Self::Available(_) => (),
        }
        self.get_available_mut().expect("just set above")
    }

    fn or_use<F>(&mut self, f: F) -> &mut InstrumentAvailable<T>
    where
        F: FnOnce() -> InstrumentAvailable<T>,
    {
        match self {
            Self::Missing => {
                *self = Self::Available(f());
            }
            Self::Available(_) => (),
        }
        self.get_available_mut().expect("just set above")
    }

    fn or_else<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce() -> Self,
    {
        match self {
            Self::Available(InstrumentAvailable::Valid(_)) => (),
            Self::Missing | Self::Available(_) => *self = f(),
        }
        self
    }

    fn outdated_or_else<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce() -> Self,
    {
        match self {
            Self::Available(InstrumentAvailable::OutOfDate(_)) => (),
            Self::Missing | Self::Available(_) => *self = f(),
        }
        self
    }

    fn replace(&mut self, t: T) -> &mut T {
        *self = Self::Available(InstrumentAvailable::Valid(t));
        self.get_mut().expect("set above")
    }

    fn or_replace<F>(&mut self, f: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        match self {
            Self::Available(InstrumentAvailable::Valid(_)) => (),
            Self::Missing | Self::Available(_) => {
                *self = Self::Available(InstrumentAvailable::Valid(f()));
            }
        }
        self.get_mut().expect("just set above")
    }

    #[expect(clippy::unused_self)]
    const fn keep(&self) {}

    fn out_of_date(&mut self) {
        let next = match self.take() {
            Self::Missing => Self::Missing,
            Self::Available(mut t) => {
                t.out_of_date();
                Self::Available(t)
            }
        };
        *self = next;
    }

    fn reset(&mut self) {
        let next = match self.take() {
            Self::Missing => Self::Missing,
            Self::Available(_) => Self::Missing,
        };
        *self = next;
    }

    fn get(&self) -> Option<&T> {
        match self {
            Self::Missing => None,
            Self::Available(t) => t.get(),
        }
    }

    fn get_any(&self) -> Option<&T> {
        match self {
            Self::Missing => None,
            Self::Available(t) => Some(t.get_any()),
        }
    }

    const fn get_available(&self) -> Option<&InstrumentAvailable<T>> {
        match self {
            Self::Missing => None,
            Self::Available(t) => Some(t),
        }
    }

    fn get_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Missing => None,
            Self::Available(t) => t.get_mut(),
        }
    }

    fn get_any_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Missing => None,
            Self::Available(t) => Some(t.get_any_mut()),
        }
    }

    const fn get_available_mut(&mut self) -> Option<&mut InstrumentAvailable<T>> {
        match self {
            Self::Missing => None,
            Self::Available(t) => Some(t),
        }
    }
}

enum InstrumentAvailable<T> {
    Transition,
    OutOfDate(T),
    Valid(T),
}

impl<T> InstrumentAvailable<T> {
    fn unwrap_outdated(self) -> T {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => t,
            Self::Valid(_) => panic!("value is valid"),
        }
    }

    fn unwrap(self) -> T {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(_) => panic!("value is out of date"),
            Self::Valid(t) => t,
        }
    }

    fn unwrap_any(self) -> T {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => t,
            Self::Valid(t) => t,
        }
    }

    fn or_update<F>(&mut self, f: F) -> &mut T
    where
        F: FnOnce(&mut T),
    {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => {
                f(t);
                let t = self.take().unwrap_outdated();
                *self = Self::Valid(t);
            }
            Self::Valid(_) => (),
        }
        self.get_mut().expect("just set to valid")
    }

    const fn take(&mut self) -> Self {
        std::mem::replace(self, Self::Transition)
    }

    fn out_of_date(&mut self) {
        *self = match self.take() {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => Self::OutOfDate(t),
            Self::Valid(t) => Self::OutOfDate(t),
        };
    }

    fn get(&self) -> Option<&T> {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(_) => None,
            Self::Valid(t) => Some(t),
        }
    }

    fn get_any(&self) -> &T {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => t,
            Self::Valid(t) => t,
        }
    }

    fn get_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(_) => None,
            Self::Valid(t) => Some(t),
        }
    }

    fn get_any_mut(&mut self) -> &mut T {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => t,
            Self::Valid(t) => t,
        }
    }
}

#[derive(Default)]
struct ImageInstruments {
    output: Instrument<PassThruTexture>,
    original: Instrument<SimpleTexture>,
    params: Instrument<DrawParameters>,

    storage_layout: Instrument<StorageSrcDstLayout>,
    kernel_layout: Instrument<KernelLayout>,
    convolution_layout: Instrument<ConvolutionPipelineLayout>,
    convolution_pipeline: Instrument<ConvolutionPipeline>,

    copy_machine: Instrument<StorageTextureCopyMachine>,
    storage_data: Instrument<SimpleStorageTexture>,
    storage_scratch: Instrument<SimpleStorageTexture>,
    kernel_bind: Instrument<KernelBinding>,
    blurred: Instrument<SimpleTexture>,

    texture_layout: Instrument<SimpleTextureLayout>,
    buffer_layout: Instrument<SimpleBufferBindLayout>,

    simple_pipeline_layout: Instrument<SimpleImageRenderPipelineLayout>,
    lanczos_pipeline_layout: Instrument<LanczosImageRenderPipelineLayout>,
    nearest_pipeline: Instrument<RenderNearestPipeline>,
    bilinear_pipeline: Instrument<RenderBilinearPipeline>,
    lanczos_pipeline: Instrument<RenderLanczosPipeline>,

    meta_buffer: Instrument<SimpleBufferBind<ImageMetadataRaw>>,
    lanczos_buffer: Instrument<SimpleBufferBind<LanczosInfoRaw>>,
}

impl ImageInstruments {
    fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    fn replaced_image(&mut self) {
        self.original.out_of_date();
        self.output.out_of_date();
        self.blurred.out_of_date();
        self.storage_data.out_of_date();
    }

    fn resized(&mut self) {
        self.output.reset();
        self.storage_data.reset();
        self.storage_scratch.reset();
        self.blurred.reset();
    }

    fn zoomed(&mut self) {
        self.output.out_of_date();
        self.blurred.out_of_date();
        self.meta_buffer.out_of_date();
    }

    fn panned(&mut self) {
        self.output.out_of_date();
        self.meta_buffer.out_of_date();
    }

    fn cycled_filter(&mut self) {
        self.output.out_of_date();
        self.original.keep();
        self.params.keep();

        self.storage_layout.reset();
        self.kernel_layout.reset();
        self.convolution_layout.reset();
        self.convolution_pipeline.reset();

        self.copy_machine.reset();
        self.storage_data.reset();
        self.storage_scratch.reset();
        self.kernel_bind.reset();
        self.blurred.reset();

        self.texture_layout.reset();
        self.buffer_layout.reset();

        self.simple_pipeline_layout.reset();
        self.lanczos_pipeline_layout.reset();
        self.nearest_pipeline.reset();
        self.bilinear_pipeline.reset();
        self.lanczos_pipeline.reset();

        self.meta_buffer.keep();
        self.lanczos_buffer.reset();
    }
}
