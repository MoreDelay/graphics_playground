use iced_wgpu::wgpu;
use iced_winit::core::Color;

use crate::{GpuContext, TargetContext};

pub struct RenderWidget {
    pipeline: wgpu::RenderPipeline,
    bg_color: Color,
}

impl RenderWidget {
    const SHADER_VERTEX: &str = "package::triangle::vert";
    const SHADER_FRAGMENT: &str = "package::triangle::frag";

    pub fn new(ctx: &GpuContext, target: &TargetContext) -> Self {
        let pipeline = Self::build_pipeline(ctx, target);
        let bg_color = Color::BLACK;
        Self { pipeline, bg_color }
    }

    pub const fn bg_color(&self) -> Color {
        self.bg_color
    }

    pub fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.draw(0..3, 0..1);
    }

    fn build_pipeline(ctx: &GpuContext, target: &TargetContext) -> wgpu::RenderPipeline {
        let vs_module = crate::gpu::create_simple_shader_module_desc(
            Some("Triangle Vertex Shader"),
            Self::SHADER_VERTEX,
        );
        let vs_module = ctx.device.create_shader_module(vs_module);
        let fs_module = crate::gpu::create_simple_shader_module_desc(
            Some("Triangle Fragment Shader"),
            Self::SHADER_FRAGMENT,
        );
        let fs_module = ctx.device.create_shader_module(fs_module);

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                push_constant_ranges: &[],
                bind_group_layouts: &[],
            });

        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                        format: target.config.format,
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
}
