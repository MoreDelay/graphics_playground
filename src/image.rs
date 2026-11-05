//! Contains the image viewer widget

pub mod filters;
mod render;

use std::path::Path;

use derive_more::Display;
use iced::Length::Fill;
use iced::{Alignment, wgpu};
use iced_widget::{button, container, pick_list, text};
use iced_winit::winit::dpi::PhysicalSize;
use iced_winit::winit::keyboard::{Key, NamedKey};
use nalgebra as na;

use crate::controls::{Message, RenderContext};
use crate::image::filters::GaussFilter;
use crate::image::render::{ImageDataInstruments, ImageMetaInstruments};
use crate::instruments::GpuContext;
use crate::instruments::bind::image::LanczosInfoRaw;
use crate::instruments::mipmap::MipMapper;
use crate::instruments::pipeline::passthru::PassThruTexture;
use crate::instruments::pipeline::{ImageFilter, ImageFilterType};
use crate::instruments::splitview::draw_splitted;
use crate::viewport::{ScrollableViewportState, ViewportMessage};

/// The image viewer widget
pub struct ImageWidget {
    /// Current meta instruments used by all images
    meta: ImageMetaInstruments,

    /// Instruments for images
    #[expect(
        clippy::vec_box,
        reason = "Large struct needs to move around inside the vec on delete, etc."
    )]
    // TODO: add feature to remove image and change image order
    images: Vec<Box<SingleImageState>>,

    /// The draw parameters used the last time
    params: DrawParameters,
    /// The current display mode
    view: ViewState,
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
            images: Vec::new(),
            params,
            view: ViewState::default(),
        }
    }

    /// Let user select new image files to load
    pub fn pick_files(&mut self) {
        let images = rfd::FileDialog::new()
            .set_title("Pick image to display")
            .add_filter("image", &["jpg", "jpeg", "png", "avif", "webp", "jxl"])
            .pick_files();
        let Some(images) = images else {
            return;
        };

        for image in images {
            let image = match Image::load(&image) {
                Ok(image) => image,
                Err(e) => {
                    eprintln!("Could not load image: {e}");
                    continue;
                }
            };
            self.add_image(image);
        }
    }

    /// Get the current image split position
    pub const fn split(&self) -> Option<ClampedSplit> {
        let got_two = self.images.len() >= 2;
        if got_two && let ViewState::Split(split) = self.view {
            let width = self.params.viewport.size().width as f32;
            Some(split.movement.clamped(width))
        } else {
            None
        }
    }

    /// Push the control widgets into the control panel column
    ///
    /// TODO: add indicator what images are displayed
    pub fn push_controls<'a>(
        &'a self,
        control: iced_widget::Column<'a, Message, iced::Theme, iced_wgpu::Renderer>,
    ) -> iced_widget::Column<'a, Message, iced::Theme, iced_wgpu::Renderer> {
        let n_images: u8 = self.images.len().try_into().expect("at most 255 images");
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
                pick_list(ImageFilter::ALL, Some(self.params.filter), |f| {
                    Message::Image(ImageMessage::SetFilter(f))
                })
                .width(Fill),
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
                    pick_list(ViewMode::ALL, Some(self.view.as_mode()), |s| {
                        Message::Image(ImageMessage::SetViewMode(s))
                    })
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

        for index in self.visible_indices() {
            let image = &mut self.images[index as usize];
            match self.params.filter.as_type() {
                ImageFilterType::Nearest => {
                    self.meta
                        .nearest(&mut image.instruments, context, &image.data, &params);
                }
                ImageFilterType::BiLinear => {
                    self.meta
                        .bilinear(&mut image.instruments, context, &image.data, &params);
                }
                ImageFilterType::Lanczos => {
                    self.meta
                        .lanczos(&mut image.instruments, context, &image.data, &params);
                }
            }
        }

        let RenderContext {
            ctx,
            passthru,
            encoder,
            viewport,
            ..
        } = context;

        let output = self.meta.create_output(ctx, passthru, viewport)?;

        match (&self.images[..], self.view) {
            ([], _) => return None,
            ([single], _) => {
                let single = single.instruments.output()?;
                passthru.full_draw(encoder, single, output.view());
            }
            (many, ViewState::Nothing) => {
                self.view = ViewState::Single(0);
                let image = &many[0];
                let image = image.instruments.output()?;
                passthru.full_draw(encoder, image, output.view());
            }
            (many, ViewState::Single(index)) => {
                let image = &many[index as usize];
                let image = image.instruments.output()?;
                passthru.full_draw(encoder, image, output.view());
            }
            (many, ViewState::Split(split)) => {
                let left = &many[split.left as usize];
                let left = left.instruments.output()?;
                let right = &many[split.right as usize];
                let right = right.instruments.output()?;
                let width = params.viewport.size().width as f32;

                draw_splitted(
                    passthru,
                    encoder,
                    viewport,
                    left,
                    right,
                    output.view(),
                    split.movement.clamped(width),
                );
            }
        }

        let output = self.meta.set_final_output(output);
        Some(output)
    }

    /// Handle a message for the image widget
    pub fn update(&mut self, message: ImageMessage) {
        match message {
            ImageMessage::Viewport(message) => self.update_viewport(message),
            ImageMessage::ResetPosition => self.reset_pos(),
            ImageMessage::SetFilter(filter) => self.set_filter(filter),
            ImageMessage::CycleFilters => self.cycle_filters(),
            ImageMessage::DragSplit { active } => self.drag_split(active),
            ImageMessage::SetViewMode(selection) => self.set_view_mode(selection),
            ImageMessage::CycleViewMode => self.cycle_view_mode(),
            ImageMessage::CycleImage { direction } => self.cycle_image(direction),
        }
    }

    /// Get a list of all indices that are currently shown
    fn visible_indices(&self) -> Vec<u8> {
        match self.view {
            ViewState::Nothing => Vec::new(),
            ViewState::Single(index) => vec![index],
            ViewState::Split(split) => vec![split.left, split.right],
        }
    }

    /// Update the image
    fn add_image(&mut self, image: Image) {
        self.images.push(Box::new(SingleImageState::new(image)));
        let new_index: u8 = (self.images.len() - 1)
            .try_into()
            .expect("at most 255 images");

        // Reset viewing settings when new image was loaded
        let movement = SplitMovement::Set(ClampedSplit::Split(0.));
        self.view = match self.view {
            ViewState::Nothing => ViewState::Single(new_index),
            ViewState::Single(index) => {
                let split = Split {
                    left: index,
                    right: new_index,
                    movement,
                };
                ViewState::Split(split)
            }
            ViewState::Split(split) => {
                let split = Split {
                    left: split.right,
                    right: new_index,
                    movement,
                };
                ViewState::Split(split)
            }
        };

        let area = self
            .visible_indices()
            .into_iter()
            .map(|index| self.images[index as usize].data.size())
            .max()
            .unwrap_or_default();
        let area = na::Vector2::new(area.width, area.height).cast();

        self.params.viewport.resize_area(area);

        // reset position of visible images
        self.meta.replaced_image();
        for index in self.visible_indices() {
            self.images[index as usize].instruments.zoomed();
        }
    }

    /// Update the viewport size
    fn resize_viewport(&mut self, size: PhysicalSize<u32>) {
        let changed = self.params.viewport.resize_view(size);
        if !changed {
            return;
        }

        self.meta.resized();
        for index in self.visible_indices() {
            self.images[index as usize].instruments.resized();
        }

        if let ViewState::Split(split) = &mut self.view
            && let SplitMovement::Set(inner) = &mut split.movement
        {
            *inner = inner.clamped(size.width as f32);
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
        for index in self.visible_indices() {
            self.images[index as usize].instruments.zoomed();
        }

        self.params.viewport.scale_zoom(factor, fix_point);
    }

    /// Handle zoom change
    fn set_zoom(&mut self, zoom: f32, fix_point: na::Point2<f32>) {
        self.meta.zoomed();
        for index in self.visible_indices() {
            self.images[index as usize].instruments.zoomed();
        }

        self.params.viewport.set_zoom(zoom, fix_point);
    }

    /// Handle panning of images
    fn pan(&mut self, pan_vector: na::Vector2<f32>) {
        self.meta.panned();

        match &mut self.view {
            ViewState::Nothing => (),
            ViewState::Single(_)
            | ViewState::Split(Split {
                movement: SplitMovement::Set(_),
                ..
            }) => {
                for index in self.visible_indices() {
                    self.images[index as usize].instruments.panned();
                }

                self.params.viewport.pan(pan_vector);
            }
            ViewState::Split(Split {
                movement: SplitMovement::Dragging(pos),
                ..
            }) => {
                // only final image is out-of-date
                *pos += pan_vector.x;
            }
        }
    }

    /// Reset the position of the images
    fn reset_pos(&mut self) {
        self.meta.panned();
        for index in self.visible_indices() {
            self.images[index as usize].instruments.panned();
        }

        self.params.viewport.reset_position();
    }

    /// Set the next filter to be used when requesting a rendering
    fn set_filter(&mut self, filter: ImageFilter) {
        self.meta.cycled_filter();
        for index in self.visible_indices() {
            self.images[index as usize].instruments.cycled_filter();
        }

        self.params.filter = filter;
    }

    /// Cycle to the next filter to be used when requesting a rendering
    fn cycle_filters(&mut self) {
        let index = ImageFilter::ALL
            .iter()
            .enumerate()
            .find_map(|(i, f)| (*f == self.params.filter).then_some(i))
            .expect("combo box has all options");
        let index = (index + 1) % ImageFilter::ALL.len();
        let filter = ImageFilter::ALL[index];
        self.set_filter(filter);
    }

    /// Update whether the split is being dragged
    const fn drag_split(&mut self, active: bool) {
        let ViewState::Split(split) = &mut self.view else {
            return;
        };

        let width = self.params.viewport.size().width as f32;
        split.movement = match active {
            true => split.movement.dragging(width),
            false => SplitMovement::Set(split.movement.clamped(width)),
        };
    }

    /// Set the view mode from a selection, preserving the current single-image index
    fn set_view_mode(&mut self, mode: ViewMode) {
        if self.view.as_mode() == mode {
            return;
        }

        self.view = match self.images.len() {
            0 => ViewState::Nothing,
            1 => ViewState::Single(0),
            n @ 2.. => match (mode, self.view) {
                (ViewMode::Single, ViewState::Split(split)) => ViewState::Single(split.left),
                (ViewMode::Split, ViewState::Single(left)) => {
                    let n: u8 = n.try_into().expect("at most 255 images");
                    let right = left.wrapping_add(1) % n;
                    let split = Split {
                        left,
                        right,
                        movement: SplitMovement::new(),
                    };
                    ViewState::Split(split)
                }
                _ => return,
            },
        };

        // Only the final composition is out-of-date, the individual image outputs remain valid
        self.meta.panned();
    }

    /// Cycle to the next view mode
    fn cycle_view_mode(&mut self) {
        // Entering single mode always shows the newest image first
        let mode = match self.view {
            ViewState::Nothing => return,
            ViewState::Single(_) => ViewMode::Split,
            ViewState::Split(_) => ViewMode::Single,
        };

        self.set_view_mode(mode);
    }

    /// Show the next or previous loaded image
    ///
    /// Only applies in [`ViewMode::Single`] mode with two images loaded.
    fn cycle_image(&mut self, direction: CycleDirection) {
        let ViewState::Single(index) = self.view else {
            return;
        };

        let index = match direction {
            CycleDirection::Next => index.checked_add(1),
            CycleDirection::Prev => index.checked_sub(1),
        };
        let Some(index) = index else { return };

        let n_images: u8 = self.images.len().try_into().expect("at most 255 images");
        let index = index.clamp(0, n_images - 1);

        self.view = ViewState::Single(index);

        // Images may have been panned between switches
        self.meta.panned();
        for index in self.visible_indices() {
            self.images[index as usize].instruments.panned();
        }
    }
}

/// State on the split view
///
/// TODO: add feature to select what image is shown on left/right, and make them swap sides
#[derive(Debug, Clone, Copy)]
struct Split {
    /// Index of the left image
    left: u8,
    /// Index of the right image
    right: u8,
    /// State about the image splitter position
    movement: SplitMovement,
}

/// State of the image splitter
#[derive(Debug, Clone, Copy)]
enum SplitMovement {
    /// Actively dragging right now
    ///
    /// The split position can be outside the bounds of the viewport
    Dragging(f32),
    /// Split is static
    Set(ClampedSplit),
}

impl SplitMovement {
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

impl Default for SplitMovement {
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
#[derive(Debug, Clone, Copy, Default)]
enum ViewState {
    /// No image is displayed
    #[default]
    Nothing,
    /// Show one image at a time; the payload is the index of the shown image
    Single(u8),
    /// Show both images side by side, split by the split bar
    Split(Split),
}

impl ViewState {
    /// The [`ViewMode`] corresponding to the current state
    const fn as_mode(self) -> ViewMode {
        match self {
            Self::Nothing => ViewMode::Single,
            Self::Single(_) => ViewMode::Single,
            Self::Split(_) => ViewMode::Split,
        }
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

impl ViewMode {
    /// All view modes in an array
    pub const ALL: [Self; 2] = [Self::Single, Self::Split];
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
#[derive(Debug, Clone, Copy)]
pub enum ImageMessage {
    /// Message for moving the contents of the viewport
    Viewport(ViewportMessage),
    /// Reset the image position in the viewport
    ResetPosition,
    /// Set image filter to this
    SetFilter(ImageFilter),
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
    const fn raw_lanczos(&self) -> LanczosInfoRaw {
        let filter_size = match self.filter {
            ImageFilter::Nearest | ImageFilter::BiLinear => 2.,
            ImageFilter::Lanczos(lobes) => lobes.get() as f32,
        };
        LanczosInfoRaw { filter_size }
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
