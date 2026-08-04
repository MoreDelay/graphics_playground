use iced::wgpu;

use crate::instruments::bind::image::{ImageMetadataRaw, LanczosInfoRaw};
use crate::instruments::bind::texture::{SimpleTexture, SimpleTextureLayout};
use crate::instruments::buffer::{SimpleBufferBind, SimpleBufferBindLayout};
use crate::instruments::pipeline::ImageFilter;
use crate::instruments::{GpuContext, SHADER_ROOT};

const SHADER_VERTEX_QUAD: &str = "package::image::quad";
const SHADER_FRAGMENT_RENDER: &str = "package::image::render";

pub struct RenderNearestPipeline(wgpu::RenderPipeline);

impl RenderNearestPipeline {
    pub fn new(
        ctx: &GpuContext,
        pipeline_layout: &SimpleImageRenderPipelineLayout,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let pipeline = image_pipeline(
            ctx,
            pipeline_layout,
            output_format,
            ImageFilter::Nearest,
            Some("Nearest Vertex Module"),
            Some("Nearest Fragment Module"),
            Some("Nearest Render Pipeline"),
        );
        Self(pipeline)
    }

    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        image: &SimpleTexture,
        meta: &SimpleBufferBind<ImageMetadataRaw>,
    ) {
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, &**image, &[]);
        pass.set_bind_group(1, &**meta, &[]);
        pass.draw(0..4, 0..1);
    }
}

pub struct RenderBilinearPipeline(wgpu::RenderPipeline);

impl RenderBilinearPipeline {
    pub fn new(
        ctx: &GpuContext,
        pipeline_layout: &SimpleImageRenderPipelineLayout,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let pipeline = image_pipeline(
            ctx,
            pipeline_layout,
            output_format,
            ImageFilter::BiLinear,
            Some("Bilinear Vertex Module"),
            Some("Bilinear Fragment Module"),
            Some("Bilinear Render Pipeline"),
        );
        Self(pipeline)
    }

    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        image: &SimpleTexture,
        meta: &SimpleBufferBind<ImageMetadataRaw>,
    ) {
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, &**image, &[]);
        pass.set_bind_group(1, &**meta, &[]);
        pass.draw(0..4, 0..1);
    }
}

pub struct RenderLanczosPipeline(wgpu::RenderPipeline);

impl RenderLanczosPipeline {
    pub fn new(
        ctx: &GpuContext,
        pipeline_layout: &LanczosImageRenderPipelineLayout,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let pipeline = image_pipeline(
            ctx,
            pipeline_layout,
            output_format,
            ImageFilter::Lanczos,
            Some("Lanczos Vertex Module"),
            Some("Lanczos Fragment Module"),
            Some("Lanczos Render Pipeline"),
        );
        Self(pipeline)
    }

    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        image: &SimpleTexture,
        meta: &SimpleBufferBind<ImageMetadataRaw>,
        lanczos: &SimpleBufferBind<LanczosInfoRaw>,
    ) {
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, &**image, &[]);
        pass.set_bind_group(1, &**meta, &[]);
        pass.set_bind_group(2, &**lanczos, &[]);
        pass.draw(0..4, 0..1);
    }
}

fn image_pipeline(
    ctx: &GpuContext,
    pipeline_layout: &wgpu::PipelineLayout,
    output_format: wgpu::TextureFormat,
    filter: ImageFilter,
    label_vertex: Option<&str>,
    label_fragment: Option<&str>,
    label_pipeline: Option<&str>,
) -> wgpu::RenderPipeline {
    let vs_module =
        crate::instruments::create_simple_shader_module_desc(label_vertex, SHADER_VERTEX_QUAD);
    let vs_module = ctx.device.create_shader_module(vs_module);

    let filter = match filter {
        ImageFilter::Nearest => "FILTER_NEAREST",
        ImageFilter::BiLinear => "FILTER_BILINEAR",
        ImageFilter::Lanczos => "FILTER_LANCZOS",
    };
    let fs_features = [(filter, true)];

    let fs_module = &SHADER_FRAGMENT_RENDER.parse().expect("module path invalid");
    let fs_module = wesl::Wesl::new(SHADER_ROOT)
        .set_features(fs_features)
        .compile(fs_module)
        .inspect_err(|e| eprintln!("WESL error: {e}"))
        .expect("shader invalid")
        .to_string();
    let fs_module = wgpu::ShaderModuleDescriptor {
        label: label_fragment,
        source: wgpu::ShaderSource::Wgsl(fs_module.into()),
    };
    let fs_module = ctx.device.create_shader_module(fs_module);

    ctx.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: label_pipeline,
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
        })
}

pub struct SimpleImageRenderPipelineLayout(wgpu::PipelineLayout);

impl SimpleImageRenderPipelineLayout {
    pub fn new(
        ctx: &GpuContext,
        texture_layout: &SimpleTextureLayout,
        buffer_layout: &SimpleBufferBindLayout,
    ) -> Self {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Image Pipeline Layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[texture_layout, buffer_layout],
            });
        Self(layout)
    }
}

impl std::ops::Deref for SimpleImageRenderPipelineLayout {
    type Target = wgpu::PipelineLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct LanczosImageRenderPipelineLayout(wgpu::PipelineLayout);

impl LanczosImageRenderPipelineLayout {
    pub fn new(
        ctx: &GpuContext,
        texture_layout: &SimpleTextureLayout,
        buffer_layout: &SimpleBufferBindLayout,
    ) -> Self {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Lanczos Image Pipeline Layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[texture_layout, buffer_layout, buffer_layout],
            });
        Self(layout)
    }
}

impl std::ops::Deref for LanczosImageRenderPipelineLayout {
    type Target = wgpu::PipelineLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
