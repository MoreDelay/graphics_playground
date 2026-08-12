use std::cell::Cell;
use std::ops::ControlFlow;
use std::path::{Path, PathBuf};

use iced::advanced::{Layout, Widget, layout, mouse, renderer, widget};
use iced::{Event, Rectangle};
use iced_wgpu::core::SmolStr;
use iced_wgpu::{Renderer, wgpu};
use iced_widget::{button, column, row, text};
use iced_winit::conversion::cursor_position;
use iced_winit::core::{Color, Element, Theme};
use iced_winit::winit::dpi::{LogicalInsets, LogicalSize, PhysicalInsets, PhysicalPosition};
use iced_winit::winit::event::{ElementState, MouseButton};
use iced_winit::winit::keyboard::ModifiersState;
use nalgebra as na;

use crate::controls::coords::{LocalCoords, LocalPoint};
use crate::image::{ComparisonSplit, ImageMemory, ImageMessage, ImageWidget};
use crate::instruments::pipeline::passthru::PassThruPipeline;
use crate::instruments::viewport::Viewport;
use crate::instruments::{GpuContext, TargetContext};
use crate::scene::RenderWidget;

pub mod coords;

#[derive(Debug, Clone)]
pub enum Message {
    SetScaleFactor(f32),
    SwitchScene,
    SelectFile,
    ScrollUp,
    ScrollDown,
    CursorMoved(na::Point2<f32>),
    MouseInput {
        button: MouseButton,
        state: ElementState,
    },
    ModifiersChanged(ModifiersState),
    KeyPress(SmolStr),
    DragSplit {
        active: bool,
    },
}

pub struct Controls {
    /// Bounds in a cell so that we can update its value with the computed layout from iced by
    /// passing a reference to the widget's draw call. The layout system gives us logical
    /// coordinates, so store them as such.
    scene_bounds: Cell<Option<PhysicalInsets<u32>>>,
    viewport: Viewport,
    passthru: PassThruPipeline,
    scene: CurrentScene,
    image: Option<PathBuf>,

    mouse_button: ElementState,
    cursor: CursorState,
    modifiers: ModifiersState,
}

impl Controls {
    pub fn new(ctx: &GpuContext, target: &TargetContext) -> Self {
        let scene_bounds = Cell::new(None);
        let viewport = Viewport::new();
        let passthru = PassThruPipeline::new(ctx, target.config.format);
        let scene = CurrentScene::scene(ctx, target);
        let image = None;
        let mouse_button = ElementState::Released;
        let cursor = CursorState::default();
        let modifiers = ModifiersState::default();
        Self {
            scene_bounds,
            viewport,
            passthru,
            scene,
            image,
            mouse_button,
            cursor,
            modifiers,
        }
    }

    pub fn view(&self, scale_factor: f32) -> Element<'_, Message, Theme, Renderer> {
        use iced::Length::{Fill, Shrink};

        self.scene_bounds.set(None);

        let bg_color = match &self.scene {
            CurrentScene::Scene(scene) => scene.bg_color(),
            CurrentScene::Image(_image) => Color::BLACK,
        };

        let bounds = &self.scene_bounds;
        let split = match &self.scene {
            CurrentScene::Scene(_) => None,
            CurrentScene::Image(image) => Some(image.split()),
        };
        let placeholder = PlaceholderWidget {
            bounds,
            split,
            bg_color,
            scale_factor,
        };
        let scene = Element::new(placeholder);

        row![
            column![
                text("Hello World").style(text::base),
                button(text("Switch").center().width(Fill))
                    .width(Fill)
                    .on_press(Message::SwitchScene),
                button(text("Pick").center().width(Fill))
                    .width(Fill)
                    .on_press(Message::SelectFile)
            ]
            .width(Shrink)
            .padding(5),
            scene
        ]
        .into()
    }

    // Handle an application-specific event
    //
    // TODO: Returns std::ops::ControlFlow because this can be short-curcuited with `?` try
    // operator. Replace with a custom type when the `Try` trait is stabilized:
    // https://github.com/rust-lang/rust/issues/84277
    pub fn update(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        message: Message,
    ) -> ControlFlow<()> {
        let cursor = match self.cursor {
            CursorState::Unknown => None,
            CursorState::LastPos(pos) => Some(pos),
        };
        let cursor = cursor.and_then(|cursor| self.viewport.coords().local_point(cursor));

        match (&mut self.scene, message) {
            (_, Message::SetScaleFactor(factor)) => self.viewport.update_scale_factor(factor),
            (_, Message::ModifiersChanged(mods)) => self.modifiers = mods,
            (_, Message::CursorMoved(position)) => self.cursor_moved(position),
            (_, Message::KeyPress(key)) => match key.as_str() {
                "q" if self.modifiers.control_key() => {
                    return ControlFlow::Break(());
                }
                _ => self.key_pressed(&key),
            },
            (_, Message::MouseInput { button, state }) => {
                let MouseButton::Left = button else {
                    return ControlFlow::Break(());
                };
                self.mouse_button = state;
            }

            (CurrentScene::Scene(_), Message::SwitchScene) => {
                self.scene = CurrentScene::image(self.image.as_deref(), &self.viewport);
            }
            (CurrentScene::Scene(_), Message::SelectFile) => {
                self.image = Self::pick_image_dialog();
                self.scene = CurrentScene::image(self.image.as_deref(), &self.viewport);
            }
            (CurrentScene::Scene(_), Message::ScrollUp) => (),
            (CurrentScene::Scene(_), Message::ScrollDown) => (),
            (CurrentScene::Scene(_), Message::DragSplit { .. }) => (),

            (CurrentScene::Image(_), Message::SwitchScene) => {
                self.scene = CurrentScene::scene(ctx, target);
            }
            (CurrentScene::Image(widget), Message::SelectFile) => {
                self.image = Self::pick_image_dialog();
                if let Some(image) = self.image.as_deref().and_then(load_image) {
                    let msg = ImageMessage::SetImage { image };
                    widget.update(msg);
                }
            }
            (CurrentScene::Image(widget), Message::ScrollUp) => {
                let message = ImageMessage::ZoomIn { cursor };
                widget.update(message);
            }
            (CurrentScene::Image(widget), Message::ScrollDown) => {
                let message = ImageMessage::ZoomOut { cursor };
                widget.update(message);
            }
            (CurrentScene::Image(widget), Message::DragSplit { active }) => {
                let message = ImageMessage::DragSplit { active };
                widget.update(message);
            }
        }
        ControlFlow::Continue(())
    }

    pub const fn min_window_size() -> LogicalSize<u32> {
        LogicalSize {
            width: 200 + 100,
            height: 200,
        }
    }

    pub const fn modifiers(&self) -> ModifiersState {
        self.modifiers
    }

    pub fn cursor(&self) -> iced::mouse::Cursor {
        let Some(pos) = self.cursor.pos() else {
            return iced::mouse::Cursor::Unavailable;
        };
        let pos = PhysicalPosition { x: pos.x, y: pos.y }.cast();
        let scale = self.viewport.coords().scale_factor();
        let cursor = cursor_position(pos, scale);
        iced::mouse::Cursor::Available(cursor)
    }

    /// Must be called after [`Controls::view`] to know the viewport bounds.
    pub fn draw_wgpu(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        view: &wgpu::TextureView,
    ) {
        let Some(bounds) = self.scene_bounds.take() else {
            eprintln!("TRIED TO DRAW WITH NO SCENE BOUNDS!");
            return;
        };

        let new_size = self.viewport.update_bounds(bounds);
        if new_size {
            let size = self.viewport.size();
            match &mut self.scene {
                CurrentScene::Scene(_) => (),
                CurrentScene::Image(image) => {
                    let msg = ImageMessage::ResizedViewport { size };
                    image.update(msg);
                }
            }
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Draw Command Encoder"),
            });

        let output = match &mut self.scene {
            CurrentScene::Scene(scene) => {
                scene.render(ctx, &self.passthru, &mut encoder, &self.viewport)
            }
            CurrentScene::Image(image) => {
                image.render(ctx, target, &self.passthru, &mut encoder, &self.viewport)
            }
        };

        let Some(output) = output else {
            return;
        };

        // self.render_to_viewport(view, &mut encoder, render_target, bounds);
        self.viewport
            .draw(&self.passthru, &mut encoder, output, view);

        ctx.queue.submit([encoder.finish()]);
    }

    fn pick_image_dialog() -> Option<PathBuf> {
        rfd::FileDialog::new()
            .set_title("Pick image to display")
            .add_filter("image", &["jpg", "jpeg", "png", "avif", "webp", "jxl"])
            .pick_file()
    }

    fn key_pressed(&mut self, key: &SmolStr) {
        let Some(pos) = self.cursor.pos() else { return };
        let pos = self.viewport.coords().local_point(pos);

        match &mut self.scene {
            CurrentScene::Scene(_) => (),
            CurrentScene::Image(widget) => {
                let Some(message) = ImageMessage::from_key(key, pos) else {
                    return;
                };
                widget.update(message);
            }
        }
    }

    fn cursor_moved(&mut self, position: na::Point2<f32>) {
        let cursor = CursorState::LastPos(position);
        let last = std::mem::replace(&mut self.cursor, cursor);

        let Some(last) = last.pos() else {
            return;
        };

        if self.mouse_button == ElementState::Released {
            return;
        }

        let offset = position - last;
        let offset = self.viewport.coords().local_vector(offset);

        match &mut self.scene {
            CurrentScene::Scene(_) => (),
            CurrentScene::Image(widget) => {
                let message = ImageMessage::Pan { offset };
                widget.update(message);
            }
        }
    }
}

#[expect(clippy::large_enum_variant)]
enum CurrentScene {
    Scene(RenderWidget),
    Image(ImageWidget),
}

impl CurrentScene {
    fn scene(ctx: &GpuContext, target: &TargetContext) -> Self {
        let scene = RenderWidget::new(ctx, target);
        Self::Scene(scene)
    }

    fn image(path: Option<&Path>, viewport: &Viewport) -> Self {
        let mut widget = ImageWidget::new();

        let size = viewport.size();
        let msg = ImageMessage::ResizedViewport { size };
        widget.update(msg);

        if let Some(image) = path.and_then(load_image) {
            let msg = ImageMessage::SetImage { image };
            widget.update(msg);
        }

        Self::Image(widget)
    }
}

#[derive(Debug, Clone)]
pub struct PlaceholderWidget<'a> {
    bounds: &'a Cell<Option<PhysicalInsets<u32>>>,
    split: Option<ComparisonSplit>,
    bg_color: Color,
    scale_factor: f32,
}

impl<Theme, Renderer> Widget<Message, Theme, Renderer> for PlaceholderWidget<'_>
where
    Renderer: renderer::Renderer,
{
    fn size(&self) -> iced::Size<iced::Length> {
        iced::Size::new(iced::Length::Fill, iced::Length::Fill)
    }

    fn update(
        &mut self,
        _tree: &mut widget::Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &Renderer,
        _clipboard: &mut dyn iced::advanced::Clipboard,
        shell: &mut iced::advanced::Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        if shell.is_event_captured() {
            return;
        }

        let Some(rect) = self.split_rect(layout) else {
            return;
        };

        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let Some(local) = self.local_cursor(layout, cursor) else {
                    return;
                };

                // TODO: use other types here, iced types should only be consumed
                let local = iced::Point {
                    x: local.x,
                    y: local.y,
                };
                let inside = rect.contains(local);
                if !inside {
                    return;
                }
                shell.capture_event();
                shell.publish(Message::DragSplit { active: true });
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                shell.publish(Message::DragSplit { active: false });
            }
            Event::Keyboard(_)
            | Event::Mouse(_)
            | Event::Window(_)
            | Event::Touch(_)
            | Event::InputMethod(_) => (),
        }
    }

    fn layout(
        &mut self,
        _tree: &mut widget::Tree,
        _renderer: &Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        layout::Node::new(limits.max())
    }

    fn draw(
        &self,
        _tree: &widget::Tree,
        renderer: &mut Renderer,
        _theme: &Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        _cursor: mouse::Cursor,
        _viewport: &iced::Rectangle,
    ) {
        // Update bounds through the cell
        let bounds = self.compute_bounds(layout);
        self.bounds.set(Some(bounds));

        // Draw the background
        renderer.fill_quad(
            renderer::Quad {
                bounds: layout.bounds(),
                ..Default::default()
            },
            self.bg_color,
        );
    }

    fn mouse_interaction(
        &self,
        _tree: &widget::Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &Renderer,
    ) -> mouse::Interaction {
        let Some(local) = self.local_cursor(layout, cursor) else {
            return mouse::Interaction::None;
        };
        let Some(rect) = self.split_rect(layout) else {
            return mouse::Interaction::None;
        };

        let local = iced::Point {
            x: local.x,
            y: local.y,
        };
        let inside = rect.contains(local);
        if !inside {
            return mouse::Interaction::None;
        }
        mouse::Interaction::Pointer
    }
}

impl PlaceholderWidget<'_> {
    const SPLIT_WIDTH: f32 = 20.;

    fn compute_bounds(&self, layout: Layout<'_>) -> PhysicalInsets<u32> {
        let bounds = layout.bounds();
        let inset = LogicalInsets {
            top: bounds.y,
            left: bounds.x,
            bottom: bounds.y + bounds.height,
            right: bounds.x + bounds.width,
        };
        inset.to_physical(self.scale_factor as f64)
    }

    fn split_rect(&self, layout: Layout<'_>) -> Option<Rectangle<f32>> {
        let bounds = self.compute_bounds(layout);
        let coords = LocalCoords::new(bounds, self.scale_factor);
        let x = match self.split? {
            ComparisonSplit::FullLeft => 0.,
            ComparisonSplit::Split(pos) => pos,
            ComparisonSplit::FullRight => coords.size().width as f32,
        };

        Some(Rectangle {
            x: x - Self::SPLIT_WIDTH / 2.,
            y: 0.,
            width: Self::SPLIT_WIDTH,
            height: coords.size().height as f32,
        })
    }

    fn local_cursor(&self, layout: Layout<'_>, cursor: iced::mouse::Cursor) -> Option<LocalPoint> {
        let iced::Point { x, y } = cursor.position()?;
        let point = na::Point2::new(x, y);

        let bounds = self.compute_bounds(layout);
        let coords = LocalCoords::new(bounds, self.scale_factor);
        coords.local_point(point)
    }
}

fn load_image(path: &Path) -> Option<ImageMemory> {
    match ImageMemory::load(path) {
        Ok(image) => return Some(image),
        Err(err) => {
            eprintln!("Could not load image: {err}");
        }
    }
    None
}

#[derive(Debug, Clone, Copy, Default)]
pub enum CursorState {
    #[default]
    Unknown,
    LastPos(na::Point2<f32>),
}

impl CursorState {
    const fn pos(self) -> Option<na::Point2<f32>> {
        match self {
            Self::Unknown => None,
            Self::LastPos(pos) => Some(pos),
        }
    }
}
