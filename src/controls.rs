use std::cell::Cell;
use std::path::{Path, PathBuf};

use iced::advanced::{Layout, Widget, layout, mouse, renderer, widget};
use iced_wgpu::{Renderer, wgpu};
use iced_widget::{button, column, row, text};
use iced_winit::core::{Color, Element, Theme};
use iced_winit::winit;

use crate::image::{ImageLoaded, ImageWidget};
use crate::scene::RenderWidget;
use crate::{GpuContext, TargetContext};

pub struct Controls {
    // Bounds in a cell so that we can update its value with the computed layout from iced by
    // passing a reference to the widget's draw call.
    scene_bounds: Cell<Option<iced::Rectangle>>,
    scene: CurrentScene,
    image: Option<PathBuf>,
}

#[derive(Debug, Clone, Copy)]
pub enum Message {
    SwitchScene,
    SelectFile,
    ScrollUp,
    ScrollDown,
    Drag { x: i32, y: i32 },
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

    pub fn update(&mut self, message: Message, ctx: &GpuContext, target: &TargetContext) {
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

            (CurrentScene::Image(_), Message::SwitchScene) => {
                self.scene = CurrentScene::scene(ctx, target);
            }
            (CurrentScene::Image(_), Message::SelectFile) => {
                self.image = Self::pick_image_dialog();
                self.scene = CurrentScene::image(self.image.as_deref(), ctx, target);
            }
            (CurrentScene::Image(widget), Message::ScrollUp) => {
                widget.zoom_in();
            }
            (CurrentScene::Image(widget), Message::ScrollDown) => {
                widget.zoom_out();
            }
            (CurrentScene::Image(widget), Message::Drag { x, y }) => {
                widget.pan(x, y);
            }
        }
    }

    pub const fn min_window_size() -> winit::dpi::PhysicalSize<u32> {
        winit::dpi::PhysicalSize {
            width: 200 + 100,
            height: 200,
        }
    }

    pub fn draw_wgpu(
        &mut self,
        ctx: &GpuContext,
        view: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let Some((mut render_pass, bounds)) = self.start_render_pass(view, encoder) else {
            return;
        };

        match &mut self.scene {
            CurrentScene::Scene(scene) => scene.draw(&mut render_pass),
            CurrentScene::Image(image) => image.draw(ctx, &mut render_pass, bounds),
        }
    }

    fn start_render_pass<'a>(
        &self,
        target: &'a wgpu::TextureView,
        encoder: &'a mut wgpu::CommandEncoder,
    ) -> Option<(wgpu::RenderPass<'a>, iced::Rectangle)> {
        let bounds = self.scene_bounds.take()?;

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: None,
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

        Some((render_pass, bounds))
    }

    fn pick_image_dialog() -> Option<PathBuf> {
        rfd::FileDialog::new()
            .add_filter("image", &["jpg", "jpeg", "png"])
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
                Ok(image) => widget.set_image(image, ctx, target),
                Err(err) => eprintln!("could not load image: {err}"),
            }
        }
        Self::Image(widget)
    }
}

#[derive(Debug, Clone)]
pub struct PlaceholderWidget<'a> {
    bounds: &'a Cell<Option<iced::Rectangle>>,
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
        self.bounds.set(Some(bounds));

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
