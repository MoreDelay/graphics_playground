use iced_wgpu::wgpu;
use iced_winit::core::Color;

use crate::instruments::pipeline::passthru::PassThruTexture;
use crate::instruments::viewport::Viewport;
use crate::instruments::{GpuContext, TargetContext};

pub struct RenderWidget {
    pipeline: wgpu::RenderPipeline,
    bg_color: Color,
    render_output: Option<PassThruTexture>,
}

impl RenderWidget {
    const SHADER_VERTEX: &str = "package::triangle::vert";
    const SHADER_FRAGMENT: &str = "package::triangle::frag";

    pub fn new(ctx: &GpuContext, target: &TargetContext) -> Self {
        let pipeline = Self::build_pipeline(ctx, target);
        let bg_color = Color::BLACK;
        let render_output = None;
        Self {
            pipeline,
            bg_color,
            render_output,
        }
    }

    pub const fn bg_color(&self) -> Color {
        self.bg_color
    }

    pub fn render(&self, encoder: &mut wgpu::CommandEncoder, output: &PassThruTexture) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Main Scene Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output.view(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        render_pass.set_pipeline(&self.pipeline);
        render_pass.draw(0..3, 0..1);
    }

    pub fn current_render_output(
        &mut self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
    ) -> Option<&PassThruTexture> {
        let last_output = self.render_output.take();
        let expected_size = viewport.extent()?;

        let next_output = if let Some(output) = last_output
            && output.texture().size() == expected_size
        {
            output
        } else {
            let output = viewport.create_texture(ctx).expect("must have size");
            self.render(encoder, &output);
            output
        };

        self.render_output = Some(next_output);
        self.render_output.as_ref()
    }

    fn build_pipeline(ctx: &GpuContext, target: &TargetContext) -> wgpu::RenderPipeline {
        let vs_module = crate::instruments::create_simple_shader_module_desc(
            Some("Triangle Vertex Shader"),
            Self::SHADER_VERTEX,
        );
        let vs_module = ctx.device.create_shader_module(vs_module);
        let fs_module = crate::instruments::create_simple_shader_module_desc(
            Some("Triangle Fragment Shader"),
            Self::SHADER_FRAGMENT,
        );
        let fs_module = ctx.device.create_shader_module(fs_module);

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Scene Render Pipeline Layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[],
            });

        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Scene Render Pipeline"),
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
