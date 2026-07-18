use std::cell::Cell;
use std::path::{Path, PathBuf};

use iced::advanced::mouse::Cursor;
use iced::advanced::{Layout, Widget, layout, mouse, renderer, widget};
use iced_wgpu::core::SmolStr;
use iced_wgpu::{Renderer, wgpu};
use iced_widget::{button, column, row, text};
use iced_winit::core::{Color, Element, Theme};
use iced_winit::winit::dpi::{LogicalInsets, LogicalSize};

use crate::gpu::{GpuContext, TargetContext};
use crate::image::{ImageLoaded, ImageMessage, ImageWidget};
use crate::scene::RenderWidget;

pub struct Controls {
    // Bounds in a cell so that we can update its value with the computed layout from iced by
    // passing a reference to the widget's draw call. The layout system gives us logical
    // coordinates, so store them as such.
    scene_bounds: Cell<Option<LogicalInsets<f32>>>,
    scene: CurrentScene,
    image: Option<PathBuf>,
}

#[derive(Debug, Clone)]
pub enum Message {
    SwitchScene,
    SelectFile,
    ScrollUp,
    ScrollDown,
    Drag(iced::Vector),
    KeyPress(SmolStr),
}

impl Controls {
    pub fn new(ctx: &GpuContext, target: &TargetContext) -> Self {
        let scene_bounds = Cell::new(None);
        let scene = CurrentScene::scene(ctx, target);
        Self {
            scene_bounds,
            scene,
            image: None,
        }
    }

    pub fn view(&self) -> Element<'_, Message, Theme, Renderer> {
        use iced::Length::{Fill, Shrink};

        self.scene_bounds.set(None);

        let bg_color = match &self.scene {
            CurrentScene::Scene(scene) => scene.bg_color(),
            CurrentScene::Image(_image) => Color::BLACK,
        };

        let bounds = &self.scene_bounds;
        let placeholder = PlaceholderWidget { bounds, bg_color };
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
        cursor: &Cursor,
    ) {
        match (&mut self.scene, message) {
            (CurrentScene::Scene(_), Message::SwitchScene) => {
                self.scene = CurrentScene::image(self.image.as_deref(), ctx, target);
            }
            (CurrentScene::Scene(_), Message::SelectFile) => {
                self.image = Self::pick_image_dialog();
                self.scene = CurrentScene::image(self.image.as_deref(), ctx, target);
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
                self.scene = CurrentScene::image(self.image.as_deref(), ctx, target);
            }
            (CurrentScene::Image(widget), Message::ScrollUp) => {
                let cursor = cursor.position();
                let message = ImageMessage::ZoomIn { cursor };
                widget.update(message);
            }
            (CurrentScene::Image(widget), Message::ScrollDown) => {
                let cursor = cursor.position();
                let message = ImageMessage::ZoomOut { cursor };
                widget.update(message);
            }
            (CurrentScene::Image(widget), Message::Drag(offset)) => {
                let message = ImageMessage::Pan { offset };
                widget.update(message);
            }
            (CurrentScene::Image(widget), Message::KeyPress(key)) => {
                let cursor = cursor.position();
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

    /// Should be called after [`Controls::view`] to know the viewport bounds.
    pub fn draw_wgpu(
        &mut self,
        ctx: &GpuContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
        scale_factor: f64,
    ) {
        let bounds = self.scene_bounds.take().unwrap_or_else(|| LogicalInsets {
            top: 0.,
            left: 0.,
            #[expect(clippy::cast_precision_loss)]
            bottom: view.texture().height() as f32,
            #[expect(clippy::cast_precision_loss)]
            right: view.texture().width() as f32,
        });

        let mut render_pass = Self::start_render_pass(view, encoder, bounds, scale_factor);
        match &mut self.scene {
            CurrentScene::Scene(scene) => scene.draw(&mut render_pass),
            CurrentScene::Image(image) => image.draw(ctx, &mut render_pass, bounds, scale_factor),
        }
    }

    fn start_render_pass<'a>(
        target: &'a wgpu::TextureView,
        encoder: &'a mut wgpu::CommandEncoder,
        bounds: LogicalInsets<f32>,
        scale_factor: f64,
    ) -> wgpu::RenderPass<'a> {
        let bounds = super::image::inset_to_rectangle(bounds, scale_factor);

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Main Image Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // iced drew the gui already, so load that
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        // limit rendering to the scene bounds
        render_pass.set_viewport(bounds.x, bounds.y, bounds.width, bounds.height, 0., 1.);
        // #[expect(clippy::cast_possible_truncation)]
        // render_pass.set_scissor_rect(
        //     bounds.x.floor() as u32,
        //     bounds.y.floor() as u32,
        //     bounds.width.ceil() as u32,
        //     bounds.height.ceil() as u32,
        // );

        render_pass
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

    fn image(path: Option<&Path>, ctx: &GpuContext, target: &TargetContext) -> Self {
        let mut widget = ImageWidget::new();
        if let Some(path) = path {
            match ImageLoaded::load(path) {
                Ok(image) => widget.set_image(ctx, target, image),
                Err(err) => eprintln!("could not load image: {err}"),
            }
        }
        Self::Image(widget)
    }
}

#[derive(Debug, Clone)]
pub struct PlaceholderWidget<'a> {
    bounds: &'a Cell<Option<LogicalInsets<f32>>>,
    bg_color: Color,
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
        self.bounds.set(Some(inset));

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
