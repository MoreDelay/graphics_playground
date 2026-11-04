//! Contains the image viewer widget

pub mod filters;
mod render;

use std::path::Path;

use derive_more::Display;
use iced::Length::Fill;
use iced::{Alignment, wgpu};
use iced_widget::{button, combo_box, container, text};
use iced_winit::winit::dpi::PhysicalSize;
use iced_winit::winit::keyboard::{Key, NamedKey};
use nalgebra as na;

use crate::controls::{Message, RenderContext};
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
    /// The current display mode
    view: ViewState,
    /// Combo box state for the image view mode selection
    combo_box_view: combo_box::State<ViewMode>,
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
        let combo_box_view = combo_box::State::new(vec![ViewMode::Single, ViewMode::Split]);

        Self {
            meta: ImageMetaInstruments::new(),
            left: None,
            right: None,
            params,
            view: ViewState::default(),
            combo_box_view,
        }
    }

    /// Get the current image split position
    pub const fn split(&self) -> Option<ClampedSplit> {
        let got_two = self.left.is_some() && self.right.is_some();
        if got_two && let ViewState::Split(split) = self.view {
            let width = self.params.viewport.size().width as f32;
            Some(split.clamped(width))
        } else {
            None
        }
    }

    /// Push the control widgets into the control panel column
    pub fn push_controls<'a>(
        &'a self,
        control: iced_widget::Column<'a, Message, iced::Theme, iced_wgpu::Renderer>,
    ) -> iced_widget::Column<'a, Message, iced::Theme, iced_wgpu::Renderer> {
        let view_ref = self
            .combo_box_view
            .options()
            .iter()
            .find(|&&m| m == self.view.as_mode())
            .expect("combo box has all modes as options");

        let n_images = self.left.is_some() as u8 + self.right.is_some() as u8;
        let got_two = n_images >= 2;
        let (active_left, active_right) = if got_two && let ViewState::Single(index) = self.view {
            (index > 0, index < n_images - 1)
        } else {
            (false, false)
        };

        let arrows = iced_widget::row![
            button(text("<").center().width(Fill))
                .width(Fill)
                .on_press_maybe(
                    active_left.then_some(Message::Image(ImageMessage::CycleImage {
                        direction: CycleDirection::Prev
                    }))
                ),
            button(text(">").center().width(Fill))
                .width(Fill)
                .on_press_maybe(
                    active_right.then_some(Message::Image(ImageMessage::CycleImage {
                        direction: CycleDirection::Next
                    }))
                ),
        ];

        let control = control
            .push(
                button(text("Open Image").center().width(Fill))
                    .width(Fill)
                    .on_press(Message::SelectFile),
            )
            .push(
                button(text("Toggle Filter").center().width(Fill))
                    .padding(5.)
                    .width(Fill)
                    .on_press(Message::Image(ImageMessage::CycleFilters)),
            )
            .push(
                button(text("Reset Pos").center().width(Fill))
                    .width(Fill)
                    .on_press(Message::Image(ImageMessage::ResetPosition)),
            )
            .push(
                button(text("Reset Zoom").center().width(Fill))
                    .width(Fill)
                    .on_press(Message::Image(ImageMessage::Viewport(
                        ViewportMessage::SetZoom {
                            fix_point: na::Point2::origin(),
                            zoom: 1.,
                        },
                    ))),
            );

        let control = match got_two {
            false => control,
            true => control
                .push(iced::widget::space().height(5.))
                .push(
                    combo_box(
                        &self.combo_box_view,
                        "Select view...",
                        Some(view_ref),
                        |selection| Message::Image(ImageMessage::SetViewMode(selection)),
                    )
                    .width(Fill),
                )
                .push(arrows),
        };

        self.push_bottom_text(control)
    }

    /// Add info text to the bottom of the control panel
    fn push_bottom_text<'a>(
        &'a self,
        control: iced_widget::Column<'a, Message, iced::Theme, iced_wgpu::Renderer>,
    ) -> iced_widget::Column<'a, Message, iced::Theme, iced_wgpu::Renderer> {
        control.push(
            container(
                text(format!(
                    "Zoom: {:.3}\nFilter: {}",
                    self.params.viewport.zoom(),
                    self.params.filter
                ))
                .style(text::base)
                .size(14)
                .align_x(Alignment::Start)
                .width(Fill),
            )
            .align_bottom(Fill),
        )
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

        match (&self.left, &self.right, self.view) {
            (None, _, _) => return None,
            (Some(left), None, _) => {
                let left = left.instruments.output()?;
                passthru.full_draw(encoder, left, output.view());
            }
            (Some(left), Some(right), ViewState::Single(index)) => {
                let image = if index == 0 { left } else { right };
                let image = image.instruments.output()?;
                passthru.full_draw(encoder, image, output.view());
            }
            (Some(left), Some(right), ViewState::Split(split)) => {
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
                    split.clamped(width),
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
            ImageMessage::SetViewMode(selection) => self.set_view_mode(selection),
            ImageMessage::CycleViewMode => self.cycle_view_mode(),
            ImageMessage::CycleImage { direction } => self.cycle_image(direction),
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

        // Reset viewing settings when new image was loaded
        let split = Split::Set(ClampedSplit::Split(0.));
        self.view = ViewState::Split(split);

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

        if let ViewState::Split(split) = &mut self.view
            && let Split::Set(inner) = split
        {
            *split = Split::Set(inner.clamped(size.width as f32));
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

        match &mut self.view {
            ViewState::Single(_) | ViewState::Split(Split::Set(_)) => {
                for image in self.iter_images_mut() {
                    image.instruments.panned();
                }

                self.params.viewport.pan(pan_vector);
            }
            ViewState::Split(Split::Dragging(pos)) => {
                // only final image is out-of-date
                *pos += pan_vector.x;
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
        let ViewState::Split(split) = &mut self.view else {
            return;
        };

        let width = self.params.viewport.size().width as f32;
        *split = match active {
            true => split.dragging(width),
            false => Split::Set(split.clamped(width)),
        };
    }

    /// Set the view mode from a selection, preserving the current single-image index
    fn set_view_mode(&mut self, selection: ViewMode) {
        if self.view.as_mode() == selection {
            return;
        }

        let next_view = match (selection, self.view) {
            (ViewMode::Single, ViewState::Split(_)) => ViewState::Single(0),
            (ViewMode::Split, ViewState::Single(_)) => ViewState::Split(Split::new()),
            _ => return,
        };

        self.view = next_view;
        // Only the final composition is out-of-date, the individual image outputs remain valid
        self.meta.panned();
    }

    /// Cycle to the next view mode
    fn cycle_view_mode(&mut self) {
        // Entering single mode always shows the newest image first
        self.view = match self.view {
            ViewState::Single(_) => ViewState::Split(Split::new()),
            ViewState::Split(_) => ViewState::Single(0),
        };
        // Only the final composition is out-of-date, the individual image outputs remain valid
        self.meta.panned();
    }

    /// Show the next or previous loaded image
    ///
    /// Only applies in [`ViewMode::Single`] mode with two images loaded.
    fn cycle_image(&mut self, direction: CycleDirection) {
        let ViewState::Single(index) = self.view else {
            return;
        };
        if self.left.is_none() || self.right.is_none() {
            return;
        }

        // TODO: have more than two images loaded at once
        let n_images = 2;
        let n_images: u8 = n_images
            .try_into()
            .expect("using u8 as index, no more than 255 images");

        let next = match direction {
            CycleDirection::Next => index.wrapping_add(1) % n_images,
            CycleDirection::Prev => index.checked_sub(1).unwrap_or(n_images - 1),
        };

        self.view = ViewState::Single(next);
        // Only the final composition is out-of-date, the individual image outputs remain valid
        self.meta.panned();
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
    /// Create the default [`Split`]
    const fn new() -> Self {
        Self::Set(ClampedSplit::Split(0.))
    }

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
        Self::new()
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

/// How the loaded images are displayed
#[derive(Debug, Clone, Copy)]
enum ViewState {
    /// Show one image at a time; the payload is the index of the shown image
    Single(u8),
    /// Show both images side by side, split by the split bar
    Split(Split),
}

impl ViewState {
    /// Create the default [`ViewMode`]
    const fn new() -> Self {
        Self::Single(0)
    }

    /// The [`ViewMode`] corresponding to the current state
    const fn as_mode(self) -> ViewMode {
        match self {
            Self::Single(_) => ViewMode::Single,
            Self::Split(_) => ViewMode::Split,
        }
    }
}

impl Default for ViewState {
    fn default() -> Self {
        Self::new()
    }
}

/// Selectable view options without payload (for the combo box)
#[derive(Debug, Clone, Copy, Display, PartialEq, Eq)]
pub enum ViewMode {
    /// Show one image at a time
    Single,
    /// Show both images side by side
    Split,
}

/// Direction to cycle between loaded images
#[derive(Debug, Clone, Copy, Display, PartialEq, Eq)]
pub enum CycleDirection {
    /// Next image
    Next,
    /// Previous image
    Prev,
}

/// The messages to update the image widget state
#[derive(Debug, Clone)]
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
    /// Select the view directly
    SetViewMode(ViewMode),
    /// Cycle through view modes
    CycleViewMode,
    /// Show the next/previous loaded image
    CycleImage {
        /// Direction of the cycle
        direction: CycleDirection,
    },
}

impl ImageMessage {
    /// Create an image message given a specific key was pressed
    pub fn from_key(key: &Key, cursor: Option<na::Point2<f32>>) -> Option<Self> {
        let fix_point = cursor.unwrap_or_default();
        #[expect(clippy::wildcard_enum_match_arm)]
        let msg = match key {
            Key::Character(char) => match char.as_str() {
                "1" => Self::Viewport(ViewportMessage::SetZoom {
                    fix_point,
                    zoom: 1.,
                }),
                "2" => Self::Viewport(ViewportMessage::SetZoom {
                    fix_point,
                    zoom: 2.,
                }),
                "9" => Self::Viewport(ViewportMessage::SetZoom {
                    fix_point,
                    zoom: 0.5,
                }),
                "s" => Self::ResetPosition,
                "f" => Self::CycleFilters,
                "v" => Self::CycleViewMode,
                "-" => Self::Viewport(ViewportMessage::zoom_out(fix_point)),
                "+" => Self::Viewport(ViewportMessage::zoom_in(fix_point)),
                _ => return None,
            },
            Key::Named(named) => match named {
                NamedKey::ArrowLeft => Self::CycleImage {
                    direction: CycleDirection::Prev,
                },
                NamedKey::ArrowRight => Self::CycleImage {
                    direction: CycleDirection::Next,
                },
                _ => return None,
            },
            Key::Unidentified(_) | Key::Dead(_) => return None,
        };
        Some(msg)
    }
}

/// Image data in memory
#[derive(Clone)]
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

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let width = self.width();
        let height = self.height();
        f.write_str(&format!("Image({width}x{height})"))
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
