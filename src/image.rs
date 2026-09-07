//! Contains the image viewer widget

pub mod filters;
mod render;

use std::path::Path;

use iced::wgpu;
use iced_wgpu::core::SmolStr;
use iced_winit::winit::dpi::PhysicalSize;
use nalgebra as na;

use crate::controls::RenderContext;
use crate::image::filters::GaussFilter;
use crate::image::render::{ImageDataInstruments, ImageMetaInstruments};
use crate::instruments::GpuContext;
use crate::instruments::bind::image::LanczosInfoRaw;
use crate::instruments::mipmap::MipMapper;
use crate::instruments::pipeline::ImageFilter;
use crate::instruments::pipeline::passthru::PassThruTexture;
use crate::instruments::splitview::draw_splitted;
use crate::viewport::{ScrollableViewportState, ViewportMessage};

/// The image viewer widget
pub struct ImageWidget {
    /// Current meta instruments used by all images
    meta: ImageMetaInstruments,

    /// Instruments for the left image
    left: Option<SingleImageState>,
    /// Instruments for the right image
    right: Option<SingleImageState>,

    /// The draw parameters used the last time
    params: DrawParameters,
    /// The state of the image split
    split: Split,
    /// Whether we are currently panning the images
    panning: bool,
}

impl ImageWidget {
    /// Create a new image viewer widget
    pub fn new() -> Self {
        let area = na::Vector2::new(1., 1.);
        let view = PhysicalSize::new(1, 1);
        let params = DrawParameters {
            viewport: ScrollableViewportState::new(area, view, 1.),
            filter: ImageFilter::default(),
        };

        Self {
            meta: ImageMetaInstruments::new(),
            left: None,
            right: None,
            params,
            split: Split::default(),
            panning: false,
        }
    }

    /// Get the current image split position
    pub const fn split(&self) -> Option<SplitReaction> {
        let got_two = self.left.is_some() && self.right.is_some();
        if got_two {
            let width = self.params.viewport.size().width as f32;
            let split = self.split.clamped(width);
            let react = matches!(self.split, Split::Dragging(..)) || !self.panning;
            let reaction = SplitReaction { split, react };
            Some(reaction)
        } else {
            None
        }
    }

    /// Get the latest rendering result
    pub fn render(&mut self, context: &mut RenderContext) -> Option<&PassThruTexture> {
        self.resize_viewport(context.viewport.size());

        if self.meta.final_output().is_some() {
            return self.meta.final_output();
        }

        let params = self.params;

        match self.params.filter {
            ImageFilter::Nearest => self.nearest(context, &params),
            ImageFilter::BiLinear => {
                self.bilinear(context, &params);
            }
            ImageFilter::Lanczos => self.lanczos(context, &params),
        }

        let RenderContext {
            ctx,
            passthru,
            encoder,
            viewport,
            ..
        } = context;

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
                let width = params.viewport.size().width as f32;
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

    /// Handle a message for the image widget
    pub fn update(&mut self, message: ImageMessage) {
        match message {
            ImageMessage::SetImage(image) => self.set_image(image),
            ImageMessage::Viewport(message) => self.update_viewport(message),
            ImageMessage::ResetPosition => self.reset_pos(),
            ImageMessage::CycleFilters => self.cycle_filters(),
            ImageMessage::DragSplit { active } => self.drag_split(active),
            ImageMessage::Panning { active } => self.panning(active),
        }
    }

    /// Render with the "nearest" filter
    fn nearest(&mut self, context: &mut RenderContext, params: &DrawParameters) {
        if let Some(image) = &mut self.left {
            self.meta
                .nearest(&mut image.instruments, context, &image.data, params);
        }

        if let Some(image) = &mut self.right {
            self.meta
                .nearest(&mut image.instruments, context, &image.data, params);
        }
    }

    /// Render with the "bilinear" filter
    fn bilinear(&mut self, context: &mut RenderContext, params: &DrawParameters) {
        if let Some(image) = &mut self.left {
            self.meta
                .bilinear(&mut image.instruments, context, &image.data, params);
        }

        if let Some(image) = &mut self.right {
            self.meta
                .bilinear(&mut image.instruments, context, &image.data, params);
        }
    }

    /// Render with the "lanczos" filter
    fn lanczos(&mut self, context: &mut RenderContext, params: &DrawParameters) {
        if let Some(image) = &mut self.left {
            self.meta
                .lanczos(&mut image.instruments, context, &image.data, params);
        }

        if let Some(image) = &mut self.right {
            self.meta
                .lanczos(&mut image.instruments, context, &image.data, params);
        }
    }

    /// Update the image
    fn set_image(&mut self, image: Image) {
        self.meta.replaced_image();
        if let Some(left) = &mut self.left {
            left.instruments.replaced_image();
        }
        if let Some(right) = &mut self.right {
            right.instruments.replaced_image();
        }

        std::mem::swap(&mut self.left, &mut self.right);
        self.left = Some(SingleImageState::new(image));

        self.split = Split::Set(ClampedSplit::Split(0.));

        let left = self.left.as_ref().map_or_default(|v| v.data.size());
        let right = self.right.as_ref().map_or_default(|v| v.data.size());
        let area = left.max(right).cast();
        let area = na::Vector2::new(area.width, area.height);

        self.params.viewport.resize_area(area);
    }

    /// Update the viewport size
    fn resize_viewport(&mut self, size: PhysicalSize<u32>) {
        let changed = self.params.viewport.resize_view(size);
        if !changed {
            return;
        }

        self.meta.resized();
        if let Some(left) = &mut self.left {
            left.instruments.resized();
        }
        if let Some(right) = &mut self.right {
            right.instruments.resized();
        }

        if let Split::Set(split) = self.split {
            self.split = Split::Set(split.clamped(size.width as f32));
        }
    }

    /// Handle a message intended for the viewport
    fn update_viewport(&mut self, message: ViewportMessage) {
        match message {
            ViewportMessage::Pan(pan) => self.pan(pan),
            ViewportMessage::SetZoom { zoom, fix_point } => self.set_zoom(zoom, fix_point),
            ViewportMessage::ScaleZoom { factor, fix_point } => self.scale_zoom(factor, fix_point),
        }
    }

    /// Handle zoom in
    fn scale_zoom(&mut self, factor: f32, fix_point: na::Point2<f32>) {
        self.meta.zoomed();
        for image in self.iter_images_mut() {
            image.instruments.zoomed();
        }

        self.params.viewport.scale_zoom(factor, fix_point);
    }

    /// Handle zoom change
    fn set_zoom(&mut self, zoom: f32, fix_point: na::Point2<f32>) {
        self.meta.zoomed();
        for image in self.iter_images_mut() {
            image.instruments.zoomed();
        }

        self.params.viewport.set_zoom(zoom, fix_point);
    }

    /// Handle panning of images
    fn pan(&mut self, pan_vector: na::Vector2<f32>) {
        self.meta.panned();

        match self.split {
            Split::Set(_) => {
                for image in self.iter_images_mut() {
                    image.instruments.panned();
                }

                self.params.viewport.pan(pan_vector);
            }
            Split::Dragging(pos) => {
                // only final image is out-of-date
                let x = pan_vector.x;
                self.split = Split::Dragging(pos + x);
            }
        }
    }

    /// Reset the position of the images
    fn reset_pos(&mut self) {
        self.meta.panned();
        if let Some(left) = &mut self.left {
            left.instruments.panned();
        }
        if let Some(right) = &mut self.right {
            right.instruments.panned();
        }

        self.params.viewport.reset_position();
    }

    /// Cycle to the next filter to be used when requesting a rendering
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

    /// Update whether the split is being dragged
    const fn drag_split(&mut self, active: bool) {
        let width = self.params.viewport.size().width as f32;
        self.split = match active {
            true => self.split.dragging(width),
            false => Split::Set(self.split.clamped(width)),
        };
    }

    /// Update whether the images are getting panned
    const fn panning(&mut self, active: bool) {
        self.panning = active;
    }

    /// Iterate over the state of all loaded images
    fn iter_images_mut(&mut self) -> impl IntoIterator<Item = &mut SingleImageState> {
        [self.left.as_mut(), self.right.as_mut()]
            .into_iter()
            .flatten()
    }
}

/// State of the image split
#[derive(Debug, Clone, Copy)]
enum Split {
    /// Actively dragging right now
    ///
    /// The split position can be outside the bounds of the viewport
    Dragging(f32),
    /// Split is static
    Set(ClampedSplit),
}

impl Split {
    /// Create the corresponding clamped split
    const fn clamped(self, width: f32) -> ClampedSplit {
        match self {
            Self::Dragging(pos) => ClampedSplit::new(pos, width),
            Self::Set(split) => split,
        }
    }

    /// Create a split that in [`Self::Dragging`] state
    const fn dragging(self, width: f32) -> Self {
        match self {
            Self::Dragging(pos) => Self::Dragging(pos),
            Self::Set(ClampedSplit::FullLeft) => Self::Dragging(-width / 2.),
            Self::Set(ClampedSplit::Split(pos)) => Self::Dragging(pos),
            Self::Set(ClampedSplit::FullRight) => Self::Dragging(width / 2.),
        }
    }
}

impl Default for Split {
    fn default() -> Self {
        Self::Set(ClampedSplit::default())
    }
}

/// The image split clamped to the current viewport sizes
#[derive(Debug, Default, Clone, Copy)]
pub enum ClampedSplit {
    /// Split is completely on the left (only right image visible)
    #[default]
    FullLeft,
    /// Both images are visible, split at the given location
    Split(f32),
    /// Split is completely on the right (only left image visible)
    FullRight,
}

impl ClampedSplit {
    /// Create a new clamped split
    const fn new(pos: f32, width: f32) -> Self {
        Self::Split(pos).clamped(width)
    }

    /// Make sure the split is clamped, respecting the provided width
    const fn clamped(self, width: f32) -> Self {
        let half = width / 2.;
        match self {
            Self::Split(pos) if pos <= -half => Self::FullLeft,
            Self::Split(pos) if pos >= half => Self::FullRight,
            keep @ (Self::Split(_) | Self::FullLeft | Self::FullRight) => keep,
        }
    }
}

/// Describes to the controller how to display the split
#[derive(Debug, Clone, Copy)]
pub struct SplitReaction {
    /// The split itself
    pub split: ClampedSplit,
    /// Whether the mouse should indicate an interaction with the split
    pub react: bool,
}

/// The rendering state for a single image
struct SingleImageState {
    /// The loaded image in memory
    data: Image,
    /// Rendering instruments for this image
    instruments: ImageDataInstruments,
}

impl SingleImageState {
    /// Create a new image state
    pub fn new(data: Image) -> Self {
        let instruments = ImageDataInstruments::new();
        Self { data, instruments }
    }
}

/// The messages to update the image widget state
#[derive(Clone)]
pub enum ImageMessage {
    /// Display a new image
    SetImage(Image),
    /// Message for moving the contents of the viewport
    Viewport(ViewportMessage),
    /// Reset the image position in the viewport
    ResetPosition,
    /// Cycle through the filters used for rendering
    CycleFilters,
    /// Start dragging the image split
    DragSplit {
        /// Whether the split is now actively dragged or no longer dragged
        active: bool,
    },
    /// Start panning the images
    Panning {
        /// Whether we are now panning or no longer panning
        active: bool,
    },
}

impl std::fmt::Debug for ImageMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SetImage(image) => f
                .debug_struct("SetImage")
                .field(
                    "image",
                    &format!("Image({}x{})", image.width(), image.height()),
                )
                .finish(),
            Self::Viewport(viewport) => f.debug_tuple("Viewport").field(viewport).finish(),
            Self::ResetPosition => write!(f, "ResetPosition"),
            Self::CycleFilters => write!(f, "CycleFilters"),
            Self::DragSplit { active } => {
                f.debug_struct("DragSplit").field("active", active).finish()
            }
            Self::Panning { active } => f.debug_struct("Panning").field("active", active).finish(),
        }
    }
}

impl ImageMessage {
    /// Create an image message given a specific key was pressed
    pub fn from_key(key: &SmolStr, cursor: Option<na::Point2<f32>>) -> Option<Self> {
        let fix_point = cursor.unwrap_or_default();
        match key.as_str() {
            "1" => Some(Self::Viewport(ViewportMessage::SetZoom {
                fix_point,
                zoom: 1.,
            })),
            "2" => Some(Self::Viewport(ViewportMessage::SetZoom {
                fix_point,
                zoom: 2.,
            })),
            "9" => Some(Self::Viewport(ViewportMessage::SetZoom {
                fix_point,
                zoom: 0.5,
            })),
            "s" => Some(Self::ResetPosition),
            "f" => Some(Self::CycleFilters),
            "-" => Some(Self::Viewport(ViewportMessage::zoom_out(fix_point))),
            "+" => Some(Self::Viewport(ViewportMessage::zoom_in(fix_point))),
            _ => None,
        }
    }
}

/// Image data in memory
#[derive(Debug, Clone)]
pub struct Image {
    /// The image's pixel data
    image: image::RgbaImage,
    /// The color format to be used during rendering
    format: wgpu::TextureFormat,
}

impl Image {
    /// The default color format for normal images
    pub const FORMAT_SRGB: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    /// Load an image from disk into memory
    pub fn load(path: &Path) -> Result<Self, image::ImageError> {
        Self::load_as(path, Self::FORMAT_SRGB)
    }

    /// Load an image from disk into memory with a specified color format
    pub fn load_as(path: &Path, format: wgpu::TextureFormat) -> Result<Self, image::ImageError> {
        let image = image::ImageReader::open(path)?
            .with_guessed_format()?
            .decode()?;
        let image = image.into();
        Ok(Self { image, format })
    }

    /// Get the size of this image
    pub fn size(&self) -> PhysicalSize<u32> {
        PhysicalSize {
            width: self.image.width(),
            height: self.image.height(),
        }
    }

    /// Get the extent of this image for use with wgpu
    pub fn extent(&self) -> wgpu::Extent3d {
        wgpu::Extent3d {
            width: self.image.width(),
            height: self.image.height(),
            depth_or_array_layers: 1,
        }
    }

    /// Upload the image to a gpu texture
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

impl std::ops::Deref for Image {
    type Target = image::RgbaImage;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

/// The parameters used to configure an image draw call
#[derive(Debug, Copy, Clone, PartialEq)]
struct DrawParameters {
    /// State about the viewport
    viewport: ScrollableViewportState,

    /// The image filter that should be applied
    ///
    /// Currently does not differentiate between magnification and minification.
    filter: ImageFilter,
}

impl DrawParameters {
    /// Create the [`LanczosInfoRaw`] corresponding to the current parameters
    #[expect(clippy::unused_self)]
    const fn raw_lanczos(&self) -> LanczosInfoRaw {
        LanczosInfoRaw { filter_size: 2. }
    }

    /// Create the blur kernel required for the current draw parameters
    fn raw_blur_kernel(&self) -> Option<Vec<f32>> {
        // This factor is a trade-off between sharpness (lower) and anti-aliasing (higher). 0.3
        // looks the best from testing around.
        const FACTOR: f32 = 0.3;

        let zoom = self.viewport.zoom();
        if zoom >= 1. {
            return None;
        }

        let sigma = FACTOR / zoom;
        let kernel = GaussFilter::new(sigma)?;
        Some(kernel.blur_kernel())
    }
}
