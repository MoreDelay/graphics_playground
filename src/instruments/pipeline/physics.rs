//! Pipelines for running 2d physics simulation

use iced::wgpu;

use crate::instruments::GpuContext;
use crate::instruments::bind::image::ViewportRaw;
use crate::instruments::buffer::{SimpleBufferBind, SimpleBufferBindLayout, VisibleVertex};
use crate::instruments::mesh::primitives::{InstanceRaw, VertexRaw};
use crate::model::MeshInstancing;

/// Physics object shaders (contains both vertex and fragment)
const SHADER_PHYSICS: &str = "package::physics::render";

/// General drawing pipeline for physics objects
pub struct PhysicsObjectPipeline(wgpu::RenderPipeline);

impl PhysicsObjectPipeline {
    /// Create a new pipeline
    pub fn new(
        ctx: &GpuContext,
        pipeline_layout: &PhysicsObjectPipelineLayout,
        output_format: wgpu::TextureFormat,
    ) -> Self {
        let module = crate::instruments::create_simple_shader_module_desc(
            Some("Physics Shader Module"),
            SHADER_PHYSICS,
        );
        let module = ctx.device.create_shader_module(module);

        let vertex_layout = VertexRaw::desc();
        let instance_layout = InstanceRaw::desc();

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Physics Render Pipeline"),
                layout: Some(pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_physics"),
                    buffers: &[vertex_layout, instance_layout],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_physics"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: output_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
            });
        Self(pipeline)
    }

    /// Draw a physics object to the scene
    pub fn draw(
        &self,
        pass: &mut wgpu::RenderPass<'_>,
        camera: &SimpleBufferBind<ViewportRaw, VisibleVertex>,
        meshes: &MeshInstancing,
    ) {
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, &**camera, &[]);

        for instancing in meshes.slice() {
            let mesh = instancing.base();
            let instances = instancing.instances();
            pass.set_vertex_buffer(0, mesh.vertices().slice(..));
            pass.set_vertex_buffer(1, instances.slice(..));
            pass.set_index_buffer(mesh.indices().slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.indices().count(), 0, 0..instances.count());
        }
    }
}

/// The physics pipeline layout
pub struct PhysicsObjectPipelineLayout(wgpu::PipelineLayout);

impl PhysicsObjectPipelineLayout {
    /// Create a new layout
    pub fn new(ctx: &GpuContext, buffer_layout: &SimpleBufferBindLayout<VisibleVertex>) -> Self {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Physics Object Pipeline Layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[buffer_layout],
            });
        Self(layout)
    }
}

impl std::ops::Deref for PhysicsObjectPipelineLayout {
    type Target = wgpu::PipelineLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
