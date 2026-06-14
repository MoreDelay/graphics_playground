use std::cell::Cell;

use iced::advanced::{Layout, Widget, layout, mouse, renderer, widget};
use iced_wgpu::{Renderer, wgpu};
use iced_winit::core::{Color, Element, Theme};

use crate::controls::Message;

#[derive(Debug, Clone)]
pub struct SceneWidget<'a> {
    bounds: &'a Cell<Option<iced::Rectangle>>,
    bg: Color,
}

impl<Message, Theme, Renderer> Widget<Message, Theme, Renderer> for SceneWidget<'_>
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
        let bounds = layout.bounds();
        self.bounds.replace(Some(bounds));

        renderer.fill_quad(
            renderer::Quad {
                bounds,
                ..Default::default()
            },
            self.bg,
        );
    }
}

pub struct Scene {
    bounds: Cell<Option<iced::Rectangle>>,
    pipeline: wgpu::RenderPipeline,
    background_color: Color,
}

impl Scene {
    pub fn new(device: &wgpu::Device, texture_format: wgpu::TextureFormat) -> Self {
        let bounds = Cell::new(None);
        let pipeline = build_pipeline(device, texture_format);
        let background_color = Color::BLACK;
        Self {
            bounds,
            pipeline,
            background_color,
        }
    }

    pub fn view(&self) -> Element<'_, Message, Theme, Renderer> {
        let bounds = &self.bounds;
        let bg = self.background_color;
        let widget = SceneWidget { bounds, bg };
        Element::new(widget)
    }

    pub fn start_render_pass<'a>(
        &self,
        target: &'a wgpu::TextureView,
        encoder: &'a mut wgpu::CommandEncoder,
    ) -> Option<wgpu::RenderPass<'a>> {
        let bounds = self.bounds.take()?;

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

        render_pass.set_viewport(bounds.x, bounds.y, bounds.width, bounds.height, 0., 1.);
        render_pass.set_scissor_rect(
            bounds.x as u32,
            bounds.y as u32,
            bounds.width as u32,
            bounds.height as u32,
        );

        Some(render_pass)
    }

    pub fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.draw(0..3, 0..1);
    }
}

fn build_pipeline(
    device: &wgpu::Device,
    texture_format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let (vs_module, fs_module) = (
        device.create_shader_module(wgpu::include_wgsl!("shader/vert.wgsl")),
        device.create_shader_module(wgpu::include_wgsl!("shader/frag.wgsl")),
    );

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        push_constant_ranges: &[],
        bind_group_layouts: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &vs_module,
            entry_point: Some("main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &fs_module,
            entry_point: Some("main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: texture_format,
                blend: Some(wgpu::BlendState {
                    color: wgpu::BlendComponent::REPLACE,
                    alpha: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
        multiview: None,
        cache: None,
    })
}
