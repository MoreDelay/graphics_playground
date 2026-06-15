use std::cell::Cell;

use iced::advanced::{Layout, Widget, layout, mouse, renderer, widget};
use iced_wgpu::{Renderer, wgpu};
use iced_widget::{button, column, row, text};
use iced_winit::core::{Color, Element, Theme};
use iced_winit::winit;

use crate::scene::Scene;

pub struct Controls {
    // Bounds in a cell so that we can update its value with the computed layout from iced by
    // passing a reference to the widget's draw call.
    scene_bounds: Cell<Option<iced::Rectangle>>,
    scene: Scene,
}

#[derive(Debug, Clone)]
pub enum Message {
    // BackgroundColorChanged(Color),
    // InputChanged(String),
}

impl Controls {
    pub const fn new(scene: Scene) -> Self {
        let scene_bounds = Cell::new(None);
        Self {
            scene_bounds,
            scene,
        }
    }

    #[expect(clippy::unused_self, clippy::needless_pass_by_ref_mut)]
    pub const fn update(&mut self, _message: Message) {
        // match message {
        //     Message::BackgroundColorChanged(color) => {
        //         self.background_color = color;
        //     }
        //     Message::InputChanged(input) => {
        //         self.input = input;
        //     }
        // }
    }

    pub fn view(&self) -> Element<'_, Message, Theme, Renderer> {
        use iced::Length::{Fill, Shrink};

        self.scene_bounds.set(None);

        let bounds = &self.scene_bounds;
        let bg_color = self.scene.bg_color();
        let placeholder = PlaceholderWidget { bounds, bg_color };
        let scene = Element::new(placeholder);

        row![
            column![
                text("Hello World").style(text::base),
                button(text("Button").center().width(Fill)).width(Fill)
            ]
            .width(Shrink)
            .padding(5),
            scene
        ]
        .into()
    }

    pub const fn min_window_size() -> winit::dpi::PhysicalSize<u32> {
        winit::dpi::PhysicalSize {
            width: 200 + 100,
            height: 200,
        }
    }

    pub fn draw_wgpu(&self, view: &wgpu::TextureView, encoder: &mut wgpu::CommandEncoder) {
        if let Some(mut render_pass) = self.start_render_pass(view, encoder) {
            self.scene.draw(&mut render_pass);
        }
    }

    fn start_render_pass<'a>(
        &self,
        target: &'a wgpu::TextureView,
        encoder: &'a mut wgpu::CommandEncoder,
    ) -> Option<wgpu::RenderPass<'a>> {
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
        render_pass.set_scissor_rect(
            bounds.x as u32,
            bounds.y as u32,
            bounds.width as u32,
            bounds.height as u32,
        );

        Some(render_pass)
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
