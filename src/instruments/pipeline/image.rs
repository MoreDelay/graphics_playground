//! Render Pipelines for the image viewer

use iced::wgpu;

use crate::instruments::bind::image::{LanczosInfoRaw, ViewportRaw};
use crate::instruments::bind::texture::{SimpleTexture, SimpleTextureLayout};
use crate::instruments::buffer::{
    SimpleBufferBind,
    SimpleBufferBindLayout,
    VisibleFragment,
    VisibleVertex,
};
use crate::instruments::mesh::InstanceBuffer;
use crate::instruments::mesh::primitives::{InstanceRaw, VertexRaw};
use crate::instruments::mesh::quad::QuadMesh;
use crate::instruments::pipeline::ImageFilter;
use crate::instruments::{GpuContext, SHADER_ROOT};

/// Image viewer vertex shader path
const SHADER_VERTEX: &str = "package::viewport";
/// Image viewer fragment shader path
const SHADER_FRAGMENT_RENDER: &str = "package::image::render";

/// Render pipeline for the "nearest" filter
pub struct RenderNearestPipeline(wgpu::RenderPipeline);

impl RenderNearestPipeline {
    /// Create a new pipeline
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

    /// Draw the image with this pipeline
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        viewport: &SimpleBufferBind<ViewportRaw, VisibleVertex>,
        image: &SimpleTexture,
        quad: &QuadMesh,
        instance: &InstanceBuffer<InstanceRaw>,
    ) {
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, &**viewport, &[]);
        pass.set_bind_group(1, &**image, &[]);

        pass.set_vertex_buffer(0, quad.vertices().slice(..));
        pass.set_vertex_buffer(1, instance.slice(..));
        pass.set_index_buffer(quad.indices().slice(..), wgpu::IndexFormat::Uint32);

        let indices = quad.indices().count();
        pass.draw_indexed(0..indices, 0, 0..1);
    }
}

/// Render pipeline for the "bilinear" filter
pub struct RenderBilinearPipeline(wgpu::RenderPipeline);

impl RenderBilinearPipeline {
    /// Create a new pipeline
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

    /// Draw the image with this pipeline
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        viewport: &SimpleBufferBind<ViewportRaw, VisibleVertex>,
        image: &SimpleTexture,
        quad: &QuadMesh,
        instance: &InstanceBuffer<InstanceRaw>,
    ) {
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, &**viewport, &[]);
        pass.set_bind_group(1, &**image, &[]);

        pass.set_vertex_buffer(0, quad.vertices().slice(..));
        pass.set_vertex_buffer(1, instance.slice(..));
        pass.set_index_buffer(quad.indices().slice(..), wgpu::IndexFormat::Uint32);

        let indices = quad.indices().count();
        pass.draw_indexed(0..indices, 0, 0..1);
    }
}

/// Render pipeline for the "lanczos" filter
pub struct RenderLanczosPipeline(wgpu::RenderPipeline);

impl RenderLanczosPipeline {
    /// Create a new pipeline
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

    /// Draw the image with this pipeline
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        viewport: &SimpleBufferBind<ViewportRaw, VisibleVertex>,
        image: &SimpleTexture,
        lanczos: &SimpleBufferBind<LanczosInfoRaw, VisibleFragment>,
        quad: &QuadMesh,
        instance: &InstanceBuffer<InstanceRaw>,
    ) {
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, &**viewport, &[]);
        pass.set_bind_group(1, &**image, &[]);
        pass.set_bind_group(2, &**lanczos, &[]);

        pass.set_vertex_buffer(0, quad.vertices().slice(..));
        pass.set_vertex_buffer(1, instance.slice(..));
        pass.set_index_buffer(quad.indices().slice(..), wgpu::IndexFormat::Uint32);

        let indices = quad.indices().count();
        pass.draw_indexed(0..indices, 0, 0..1);
    }
}

/// Create an image pipeline
///
/// As the creation is very similar, this function handles the creation for all pipelines,
/// parameterized by the `filter` argument.
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
        crate::instruments::create_simple_shader_module_desc(label_vertex, SHADER_VERTEX);
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

    let vertex_layout = VertexRaw::desc();
    let instance_layout = InstanceRaw::desc();

    ctx.device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: label_pipeline,
            layout: Some(pipeline_layout),
            vertex: wgpu::VertexState {
                module: &vs_module,
                entry_point: Some("vs_viewport"),
                buffers: &[vertex_layout, instance_layout],
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

/// A simple image render pipeline layout
///
/// Used by [`RenderNearestPipeline`] and [`RenderBilinearPipeline`].
pub struct SimpleImageRenderPipelineLayout(wgpu::PipelineLayout);

impl SimpleImageRenderPipelineLayout {
    /// Create a new layout
    pub fn new(
        ctx: &GpuContext,
        buffer_layout: &SimpleBufferBindLayout<VisibleVertex>,
        texture_layout: &SimpleTextureLayout,
    ) -> Self {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Image Pipeline Layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[buffer_layout, texture_layout],
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

/// The image render pipeline layout for using the lanczos filter
pub struct LanczosImageRenderPipelineLayout(wgpu::PipelineLayout);

impl LanczosImageRenderPipelineLayout {
    /// Create a new layout
    pub fn new(
        ctx: &GpuContext,
        texture_layout: &SimpleTextureLayout,
        buffer_layout_vertex: &SimpleBufferBindLayout<VisibleVertex>,
        buffer_layout_fragment: &SimpleBufferBindLayout<VisibleFragment>,
    ) -> Self {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Lanczos Image Pipeline Layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[buffer_layout_vertex, texture_layout, buffer_layout_fragment],
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
