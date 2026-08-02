use std::cell::Cell;
use std::path::{Path, PathBuf};

use iced::advanced::{Layout, Widget, layout, mouse, renderer, widget};
use iced_wgpu::core::SmolStr;
use iced_wgpu::{Renderer, wgpu};
use iced_widget::{button, column, row, text};
use iced_winit::core::{Color, Element, Theme};
use iced_winit::winit::dpi::{LogicalInsets, LogicalSize, PhysicalInsets};

use crate::image::{ImageLoaded, ImageMessage, ImageWidget};
use crate::instruments::viewport::{VPPoint, VPVector, Viewport};
use crate::instruments::{GpuContext, TargetContext};
use crate::scene::RenderWidget;

#[derive(Debug, Clone)]
pub enum Message {
    SwitchScene,
    SelectFile,
    ScrollUp,
    ScrollDown,
    Drag(VPVector),
    KeyPress(SmolStr),
}

pub struct Controls {
    /// Bounds in a cell so that we can update its value with the computed layout from iced by
    /// passing a reference to the widget's draw call. The layout system gives us logical
    /// coordinates, so store them as such.
    scene_bounds: Cell<Option<PhysicalInsets<u32>>>,
    viewport: Viewport,
    scene: CurrentScene,
    image: Option<PathBuf>,
}

impl Controls {
    pub fn new(ctx: &GpuContext, target: &TargetContext) -> Self {
        let scene_bounds = Cell::new(None);
        let viewport = Viewport::new(ctx, target.config.format);
        let scene = CurrentScene::scene(ctx, target);
        let image = None;
        Self {
            scene_bounds,
            viewport,
            scene,
            image,
        }
    }

    pub fn view(&self, scale_factor: f64) -> Element<'_, Message, Theme, Renderer> {
        use iced::Length::{Fill, Shrink};

        self.scene_bounds.set(None);

        let bg_color = match &self.scene {
            CurrentScene::Scene(scene) => scene.bg_color(),
            CurrentScene::Image(_image) => Color::BLACK,
        };

        let bounds = &self.scene_bounds;
        let placeholder = PlaceholderWidget {
            bounds,
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

    pub fn update(
        &mut self,
        message: Message,
        ctx: &GpuContext,
        target: &TargetContext,
        cursor: Option<VPPoint>,
    ) {
        match (&mut self.scene, message) {
            (CurrentScene::Scene(_), Message::SwitchScene) => {
                self.scene = CurrentScene::image(self.image.as_deref());
            }
            (CurrentScene::Scene(_), Message::SelectFile) => {
                self.image = Self::pick_image_dialog();
                self.scene = CurrentScene::image(self.image.as_deref());
            }
            (CurrentScene::Scene(_), Message::ScrollUp) => (),
            (CurrentScene::Scene(_), Message::ScrollDown) => (),
            (CurrentScene::Scene(_), Message::Drag { .. }) => (),
            (CurrentScene::Scene(_), Message::KeyPress(..)) => (),

            (CurrentScene::Image(_), Message::SwitchScene) => {
                self.scene = CurrentScene::scene(ctx, target);
            }
            (CurrentScene::Image(_), Message::SelectFile) => {
                self.image = Self::pick_image_dialog();
                self.scene = CurrentScene::image(self.image.as_deref());
            }
            (CurrentScene::Image(widget), Message::ScrollUp) => {
                let message = ImageMessage::ZoomIn { cursor };
                widget.update(message);
            }
            (CurrentScene::Image(widget), Message::ScrollDown) => {
                let message = ImageMessage::ZoomOut { cursor };
                widget.update(message);
            }
            (CurrentScene::Image(widget), Message::Drag(offset)) => {
                let message = ImageMessage::Pan { offset };
                widget.update(message);
            }
            (CurrentScene::Image(widget), Message::KeyPress(key)) => {
                let Some(message) = ImageMessage::from_key(&key, cursor) else {
                    return;
                };
                widget.update(message);
            }
        }
    }

    pub const fn min_window_size() -> LogicalSize<u32> {
        LogicalSize {
            width: 200 + 100,
            height: 200,
        }
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

        self.viewport.resize(bounds);

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Frame Draw Command Encoder"),
            });

        let output = match &mut self.scene {
            CurrentScene::Scene(scene) => {
                scene.current_render_output(ctx, &mut encoder, &self.viewport)
            }
            CurrentScene::Image(image) => image.render(ctx, target, &mut encoder, &self.viewport),
        };

        let Some(output) = output else {
            return;
        };

        // self.render_to_viewport(view, &mut encoder, render_target, bounds);
        self.viewport.draw(&mut encoder, output, view);

        ctx.queue.submit([encoder.finish()]);
    }

    fn pick_image_dialog() -> Option<PathBuf> {
        rfd::FileDialog::new()
            .add_filter("image", &["jpg", "jpeg", "png", "avif", "webp", "jxl"])
            .pick_file()
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

    fn image(path: Option<&Path>) -> Self {
        println!("image scene");
        let mut widget = ImageWidget::new();
        if let Some(path) = path {
            match ImageLoaded::load(path) {
                Ok(image) => {
                    let msg = ImageMessage::SetImage { image };
                    widget.update(msg);
                }
                Err(err) => eprintln!("could not load image: {err}"),
            }
        }
        Self::Image(widget)
    }
}

#[derive(Debug, Clone)]
pub struct PlaceholderWidget<'a> {
    bounds: &'a Cell<Option<PhysicalInsets<u32>>>,
    bg_color: Color,
    scale_factor: f64,
}

impl<Message, Theme, Renderer> Widget<Message, Theme, Renderer> for PlaceholderWidget<'_>
where
    Renderer: renderer::Renderer,
{
    fn size(&self) -> iced::Size<iced::Length> {
        iced::Size::new(iced::Length::Fill, iced::Length::Fill)
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
        let bounds = layout.bounds();
        let inset = LogicalInsets {
            top: bounds.y,
            left: bounds.x,
            bottom: bounds.y + bounds.height,
            right: bounds.x + bounds.width,
        };
        self.bounds.set(Some(inset.to_physical(self.scale_factor)));

        // Draw the background
        renderer.fill_quad(
            renderer::Quad {
                bounds,
                ..Default::default()
            },
            self.bg_color,
        );
    }
}
