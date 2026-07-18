use iced::wgpu;

use crate::gpu::bind::{
    ImageMetadataBind,
    ImageMetadataLayout,
    SingleTextureBind,
    SingleTextureLayout,
};
use crate::gpu::{GpuContext, SHADER_ROOT};

const SHADER_VERTEX_QUAD: &str = "package::image::quad";
const SHADER_FRAGMENT_RENDER: &str = "package::image::render";

pub struct ImageRenderPipelines {
    nearest: RenderNearestPipeline,
    bilinear: RenderBilinearPipeline,
    output_format: wgpu::TextureFormat,
}

impl ImageRenderPipelines {
    pub fn new(ctx: &GpuContext, output_format: wgpu::TextureFormat) -> Self {
        let texture_layout = SingleTextureLayout::new(ctx);
        let meta_layout = ImageMetadataLayout::new(ctx);

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Image Pipeline Layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[&texture_layout, &meta_layout],
            });
        let nearest = RenderNearestPipeline::new(ctx, &pipeline_layout, output_format);
        let bilinear = RenderBilinearPipeline::new(ctx, &pipeline_layout, output_format);

        Self {
            nearest,
            bilinear,
            output_format,
        }
    }

    #[expect(unused)]
    pub const fn output(&self) -> wgpu::TextureFormat {
        self.output_format
    }

    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        choice: PipelineChoice,
        image: &SingleTextureBind,
        meta: &ImageMetadataBind,
    ) {
        match choice {
            PipelineChoice::Nearest => pass.set_pipeline(&self.nearest.0),
            PipelineChoice::Bilinear => pass.set_pipeline(&self.bilinear.0),
        }
        pass.set_bind_group(0, &**image, &[]);
        pass.set_bind_group(1, &**meta, &[]);
        pass.draw(0..4, 0..1);
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PipelineChoice {
    Nearest,
    Bilinear,
}

struct RenderNearestPipeline(wgpu::RenderPipeline);

impl RenderNearestPipeline {
    fn new(
        ctx: &GpuContext,
        pipeline_layout: &wgpu::PipelineLayout,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let vs_module =
            crate::gpu::create_simple_shader_module_desc(Some("Quad Shader"), SHADER_VERTEX_QUAD);
        let vs_module = ctx.device.create_shader_module(vs_module);

        let fs_module = &SHADER_FRAGMENT_RENDER.parse().expect("module path invalid");
        let fs_module = wesl::Wesl::new(SHADER_ROOT)
            .compile(fs_module)
            .inspect_err(|e| eprintln!("WESL error: {e}"))
            .expect("shader invalid")
            .to_string();
        let fs_module = wgpu::ShaderModuleDescriptor {
            label: Some("ColorNormalShader"),
            source: wgpu::ShaderSource::Wgsl(fs_module.into()),
        };
        let fs_module = ctx.device.create_shader_module(fs_module);

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Image Nearest Pipeline"),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vs_module,
                    entry_point: Some("vs_quad"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs_module,
                    entry_point: Some("fs_image"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: output_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
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
            });

        Self(pipeline)
    }
}

struct RenderBilinearPipeline(wgpu::RenderPipeline);

impl RenderBilinearPipeline {
    const SHADER_VERTEX: &str = "package::image::quad";
    const SHADER_FRAGMENT: &str = "package::image::render";

    fn new(
        ctx: &GpuContext,
        pipeline_layout: &wgpu::PipelineLayout,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let vs_module =
            crate::gpu::create_simple_shader_module_desc(Some("Quad Shader"), Self::SHADER_VERTEX);
        let vs_module = ctx.device.create_shader_module(vs_module);

        let fs_features = [("FILTER_BILINEAR", true)];
        let fs_module = &Self::SHADER_FRAGMENT.parse().expect("module path invalid");
        let fs_module = wesl::Wesl::new(SHADER_ROOT)
            .set_features(fs_features)
            .compile(fs_module)
            .inspect_err(|e| eprintln!("WESL error: {e}"))
            .expect("shader invalid")
            .to_string();
        let fs_module = wgpu::ShaderModuleDescriptor {
            label: Some("ColorNormalShader"),
            source: wgpu::ShaderSource::Wgsl(fs_module.into()),
        };
        let fs_module = ctx.device.create_shader_module(fs_module);

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Image Pipeline"),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vs_module,
                    entry_point: Some("vs_quad"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs_module,
                    entry_point: Some("fs_image"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: output_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
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
            });

        Self(pipeline)
    }
}
