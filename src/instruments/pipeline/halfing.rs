//! Instruments for downsampling textures by half

use iced::wgpu;

use crate::instruments::GpuContext;
use crate::instruments::bind::storage::{SimpleStorageTexture, StorageSrcDstLayout};

/// The downsampling pipeline, halfing the texture in each dimension
pub struct HalfingPipeline(wgpu::ComputePipeline);

impl HalfingPipeline {
    /// Halfing compute shader path
    const SHADER_HALFING: &str = "package::mipmap::halfing";

    /// Create a new pipeline
    pub fn new(
        ctx: &GpuContext,
        layout: &HalfingPipelineLayout,
        label_shader: Option<&str>,
        label_pipeline: Option<&str>,
    ) -> Self {
        let module = crate::instruments::create_simple_shader_module_desc(
            label_shader,
            Self::SHADER_HALFING,
        );
        let module = ctx.device.create_shader_module(module);
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: label_pipeline,
                layout: Some(layout),
                module: &module,
                entry_point: Some("halfing"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self(pipeline)
    }

    /// Execute the downsampling
    pub fn run(
        &self,
        ctx: &GpuContext,
        pass: &mut wgpu::ComputePass,
        storage_layout: &StorageSrcDstLayout,
        storage_src: &SimpleStorageTexture,
        storage_dst: &SimpleStorageTexture,
        source_mip_level: u32,
    ) {
        let target_mip_level = source_mip_level + 1;

        let src_view = storage_src.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: source_mip_level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dst_view = storage_dst.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: target_mip_level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Halfing BindGroup"),
            layout: storage_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
            ],
        });

        // divide by 2^mip_level
        let dispatch_x = storage_dst.width() >> target_mip_level;
        let dispatch_y = storage_dst.height() >> target_mip_level;
        let dispatch_x = dispatch_x.div_ceil(16);
        let dispatch_y = dispatch_y.div_ceil(16);

        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, &texture_bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}

/// The layout for [`HalfingPipeline`]
pub struct HalfingPipelineLayout(wgpu::PipelineLayout);

impl HalfingPipelineLayout {
    /// Create a new layout
    pub fn new(ctx: &GpuContext, storage: &StorageSrcDstLayout, label: Option<&str>) -> Self {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label,
                bind_group_layouts: &[storage],
                push_constant_ranges: &[],
            });
        Self(layout)
    }
}

impl std::ops::Deref for HalfingPipelineLayout {
    type Target = wgpu::PipelineLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
