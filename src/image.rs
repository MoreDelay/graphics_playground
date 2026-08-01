mod filters;
mod render;

use std::path::Path;

use iced::wgpu;
use iced_wgpu::core::SmolStr;
use iced_winit::winit::dpi::PhysicalSize;
use nalgebra as na;

use crate::gpu::bind::{
    ImageMetadataBind,
    ImageMetadataLayout,
    ImageMetadataRaw,
    SingleTextureBind,
    SingleTextureLayout,
};
use crate::gpu::pipeline::{ImageRenderPipelines, PassThruTexture, PipelineChoice};
use crate::gpu::viewport::{VPPoint, VPVector, Viewport};
use crate::gpu::{GpuContext, SimpleBuffer, TargetContext};
use crate::image::render::ImageFilter;
use crate::image::render::lanczos::Interpolator;
use crate::image::render::mipmap::MipMapper;

pub struct ImageWidget {
    data: Option<WidgetState>,
}

impl ImageWidget {
    pub const fn new() -> Self {
        let data = None;
        Self { data }
    }

    pub fn set_image(&mut self, ctx: &GpuContext, target: &TargetContext, image: ImageLoaded) {
        let data = WidgetState::new(ctx, target, image);
        self.data = Some(data);
    }

    pub fn current_render_output(
        &mut self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
    ) -> Option<&PassThruTexture> {
        let data = self.data.as_mut()?;
        data.current_render_output(ctx, encoder, viewport)
    }

    pub fn update(&mut self, message: ImageMessage) {
        if let Some(data) = &mut self.data {
            data.update(message);
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ImageMessage {
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

    pub fn upload(&self, ctx: &GpuContext) -> wgpu::Texture {
        let size = wgpu::Extent3d {
            width: self.image.width(),
            height: self.image.height(),
            depth_or_array_layers: 1,
        };

        let mip_level_count = size.width.min(size.height).ilog2() + 1;
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image texture"),
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
    image: ImageLoaded,
    pipelines: ImageRenderPipelines,
    render_state: Option<ImageRenderState>,
    render_output: Option<PassThruTexture>,

    // persistent state
    offset: na::Vector2<f32>,
    zoom: f32,
    filter: ImageFilter,
}

impl WidgetState {
    const SCALE_INCREASE_FACTOR: f32 = 1.2;
    const ZOOM_MAX: f32 = 100.0;
    const ZOOM_MIN: f32 = 0.05;

    pub fn new(ctx: &GpuContext, target: &TargetContext, image: ImageLoaded) -> Self {
        let pipelines = ImageRenderPipelines::new(ctx, target.config.format);
        let offset = na::Vector2::default();
        Self {
            image,
            pipelines,
            render_state: None,
            render_output: None,
            offset,
            zoom: 1.,
            filter: ImageFilter::Nearest,
        }
    }

    pub fn render(
        &mut self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        output: &PassThruTexture,
        draw_data: DrawData,
    ) {
        let render = if let Some(render) = self.render_state.as_mut() {
            render.update(ctx, draw_data);
            render
        } else {
            self.render_state = Some(ImageRenderState::new(ctx, &self.image, draw_data));
            self.render_state.as_mut().expect("just set above")
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Main Image Render Pass"),
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

        match self.filter {
            ImageFilter::Nearest => self.pipelines.draw(
                &mut pass,
                PipelineChoice::Nearest,
                &render.original_bind,
                &render.buffer_bind,
            ),
            ImageFilter::BiLinear => self.pipelines.draw(
                &mut pass,
                PipelineChoice::Bilinear,
                &render.original_bind,
                &render.buffer_bind,
            ),
            ImageFilter::Lanczos => {
                let prepared = &render
                    .prepared
                    .as_ref()
                    .expect("must be set by update above")
                    .bind;
                self.pipelines.draw(
                    &mut pass,
                    PipelineChoice::Nearest,
                    prepared,
                    &render.buffer_bind,
                );
            }
        }
    }

    pub fn current_render_output(
        &mut self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
    ) -> Option<&PassThruTexture> {
        let last_output = self.render_output.take();
        let expected_size = viewport.extent()?;

        let next_output = if let Some(output) = last_output
            && output.texture().size() == expected_size
        {
            output
        } else {
            viewport.create_texture(ctx)?
        };

        let draw_data = self.create_draw_data(viewport.size()?);
        match self.render_state.as_ref() {
            Some(render_state) if render_state.basis == draw_data => {}
            _ => self.render(ctx, encoder, &next_output, draw_data),
        }

        self.render_output = Some(next_output);
        self.render_output.as_ref()
    }

    pub fn update(&mut self, message: ImageMessage) {
        match message {
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

    fn zoom_in(&mut self, fix_point: VPPoint) {
        let zoom = self.zoom * Self::SCALE_INCREASE_FACTOR;
        self.set_zoom(zoom, fix_point);
    }

    fn zoom_out(&mut self, fix_point: VPPoint) {
        let zoom = self.zoom / Self::SCALE_INCREASE_FACTOR;
        self.set_zoom(zoom, fix_point);
    }

    fn set_zoom(&mut self, zoom: f32, fix_point: VPPoint) {
        let zoom = zoom.clamp(Self::ZOOM_MIN, Self::ZOOM_MAX);

        // get offset in fix-point coordinates (where fix-point is the origin)
        let offset = self.offset - fix_point.coords;

        // scale up offset position by actual difference of scale factor
        let factor = zoom / self.zoom;
        let offset = offset * factor;

        // return back to viewport coordinates
        let offset = offset + fix_point.coords;

        self.offset = offset;
        self.zoom = zoom;
        println!("zoom: {zoom}");

        // when the image is at the border, it might move out of frame by zooming
        self.clamp_offset();
    }

    fn pan(&mut self, offset: VPVector) {
        self.offset.x += offset.x;
        self.offset.y += offset.y;
        self.clamp_offset();
    }

    fn reset_pos(&mut self) {
        self.offset = na::Vector2::zeros();
    }

    fn cycle_filters(&mut self) {
        self.filter = match self.filter {
            render::ImageFilter::Nearest => render::ImageFilter::BiLinear,
            render::ImageFilter::BiLinear => render::ImageFilter::Lanczos,
            render::ImageFilter::Lanczos => render::ImageFilter::Nearest,
        };
        println!("Filter: {:?}", self.filter);
    }

    /// Make sure that at least 10% of the viewport area shows part of the image.
    fn clamp_offset(&mut self) {
        const FILLED_PERCENT: f32 = 0.1;

        let Some(render) = self.render_state.as_ref() else {
            self.offset = na::Vector2::zeros();
            return;
        };
        let viewport = render.basis.viewport;

        let size = self.image.size();

        #[expect(clippy::cast_precision_loss)]
        let width = viewport.width as f32;
        #[expect(clippy::cast_precision_loss)]
        let height = viewport.height as f32;

        #[expect(clippy::cast_precision_loss)]
        let x_min = width.mul_add(FILLED_PERCENT, -self.zoom * size.width as f32);
        let x_max = width * (1. - FILLED_PERCENT);
        let x = self.offset.x.clamp(x_min, x_max);

        #[expect(clippy::cast_precision_loss)]
        let y_min = height.mul_add(FILLED_PERCENT, -self.zoom * size.height as f32);
        let y_max = height * (1. - FILLED_PERCENT);
        let y = self.offset.y.clamp(y_min, y_max);

        self.offset = na::Vector2::new(x, y);
    }

    fn create_draw_data(&self, viewport: PhysicalSize<u32>) -> DrawData {
        DrawData {
            viewport,
            offset: self.offset,
            size: self.image.size(),
            zoom: self.zoom,
            filter: self.filter,
        }
    }
}

struct ImageRenderState {
    basis: DrawData,
    original: wgpu::Texture,
    original_bind: SingleTextureBind,
    prepared: Option<PreparedImage>,

    metadata_buffer: SimpleBuffer<ImageMetadataRaw>,
    buffer_bind: ImageMetadataBind,

    lanczos: Interpolator,
}

impl ImageRenderState {
    fn new(ctx: &GpuContext, image: &ImageLoaded, basis: DrawData) -> Self {
        let texture_layout = SingleTextureLayout::new(ctx);
        let original = image.upload(ctx);
        let original_bind = SingleTextureBind::new(ctx, &texture_layout, &original);

        let metadata_buffer = basis.raw_metadata();
        let metadata_buffer =
            SimpleBuffer::new(ctx, metadata_buffer, Some("Image Metadata Buffer"));

        let buffer_bind_layout = ImageMetadataLayout::new(ctx);
        let buffer_bind = ImageMetadataBind::new(ctx, &buffer_bind_layout, &metadata_buffer);

        let lanczos = render::lanczos::Interpolator::new(ctx);

        Self {
            basis,
            original,
            original_bind,
            prepared: None,
            metadata_buffer,
            buffer_bind,
            lanczos,
        }
    }

    fn update(&mut self, ctx: &GpuContext, data: DrawData) {
        if self.basis == data {
            return;
        }

        match (data.filter, &self.prepared) {
            (ImageFilter::Nearest | ImageFilter::BiLinear, _) => {
                self.prepared = None;

                self.metadata_buffer.update(ctx, data.raw_metadata());
            }
            (ImageFilter::Lanczos, None) => {
                self.prepared = Some(PreparedImage::lanczos(
                    ctx,
                    &self.lanczos,
                    data.zoom,
                    &self.original,
                ));

                let mut meta = data.raw_metadata();
                meta.zoom = 1.;
                self.metadata_buffer.update(ctx, meta);
            }
            (ImageFilter::Lanczos, Some(PreparedImage { zoom, .. })) if data.zoom.ne(zoom) => {
                self.prepared = Some(PreparedImage::lanczos(
                    ctx,
                    &self.lanczos,
                    data.zoom,
                    &self.original,
                ));

                let mut meta = data.raw_metadata();
                meta.zoom = 1.;
                self.metadata_buffer.update(ctx, meta);
            }
            (ImageFilter::Lanczos, Some(_)) => {
                let mut meta = data.raw_metadata();
                meta.zoom = 1.;
                self.metadata_buffer.update(ctx, meta);
            }
        }
        self.basis = data;
    }
}

struct PreparedImage {
    #[expect(unused)]
    texture: wgpu::Texture,
    bind: SingleTextureBind,

    zoom: f32,
    // filter: ImageFilter,
}

impl PreparedImage {
    fn lanczos(
        ctx: &GpuContext,
        interpolator: &Interpolator,
        zoom: f32,
        original: &wgpu::Texture,
    ) -> Self {
        #[expect(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let width = ((original.width() as f32) * zoom) as u32;
        #[expect(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
        let height = ((original.height() as f32) * zoom) as u32;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Prepared Lanczos Interpolated Image Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: original.format(),
            usage: original.usage(),
            view_formats: &[],
        });
        interpolator.filter(ctx, original, &texture);

        let layout = SingleTextureLayout::new(ctx);
        let bind = SingleTextureBind::new(ctx, &layout, &texture);
        Self {
            texture,
            bind,
            zoom,
            // filter: ImageFilter::Lanczos,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DrawData {
    /// Widget location and size as determined by iced layout.
    ///
    /// Needed to transform window to widget coordinates.
    viewport: PhysicalSize<u32>,
    /// Image starts at this offset from the top left corner of the viewport.
    offset: na::Vector2<f32>,
    /// Size of the image.
    size: PhysicalSize<u32>,
    /// Image is scaled by this factor.
    zoom: f32,
    /// The image filter that should be applied
    ///
    /// Currently does not differentiate between magnification and minification.
    filter: render::ImageFilter,
}

impl DrawData {
    fn raw_metadata(&self) -> ImageMetadataRaw {
        ImageMetadataRaw {
            start: [self.offset.x, self.offset.y],
            zoom: self.zoom,
            _pad: 0,
        }
    }
}
