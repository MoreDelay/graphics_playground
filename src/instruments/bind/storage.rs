use std::range::Range;

use iced::wgpu;

use crate::instruments::GpuContext;

pub const FORMAT_STORAGE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

pub struct SimpleStorageTexture(wgpu::Texture);

impl SimpleStorageTexture {
    pub fn empty(ctx: &GpuContext, base: &wgpu::Texture, label: Option<&str>) -> Self {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: base.size(),
            mip_level_count: base.mip_level_count(),
            sample_count: base.sample_count(),
            dimension: base.dimension(),
            format: FORMAT_STORAGE,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        Self(texture)
    }

    pub fn copy_to_texture(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        machine: &StorageTextureCopyMachine,
        dst: &wgpu::Texture,
        mip_range: Range<u32>,
    ) {
        for mip_level in mip_range {
            machine.to_texture(ctx, encoder, &self.0, dst, mip_level);
        }
    }

    pub fn copy_from_texture(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        machine: &StorageTextureCopyMachine,
        src: &wgpu::Texture,
        mip_range: Range<u32>,
    ) {
        for mip_level in mip_range {
            machine.to_storage(ctx, encoder, src, &self.0, mip_level);
        }
    }
}

impl std::ops::Deref for SimpleStorageTexture {
    type Target = wgpu::Texture;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct StorageTextureCopyMachine {
    texture_layout: wgpu::BindGroupLayout,
    texture_to_storage: wgpu::RenderPipeline,
    storage_to_texture: wgpu::RenderPipeline,
}

impl StorageTextureCopyMachine {
    const SHADER_COPY_VERTEX: &str = "package::image::quad";
    const SHADER_COPY_FRAGMENT: &str = "package::mipmap::texture_copy";

    pub fn new(ctx: &GpuContext, original_format: wgpu::TextureFormat) -> Self {
        let texture_layout = Self::create_texture_layout(ctx);
        let (texture_to_storage, storage_to_texture) =
            Self::create_copy_pipelines(ctx, &texture_layout, original_format);

        Self {
            texture_layout,
            texture_to_storage,
            storage_to_texture,
        }
    }

    pub fn to_storage(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        mip_level: u32,
    ) {
        assert_eq!(src.size(), dst.size());
        self.run_internal(ctx, encoder, src, dst, mip_level, &self.texture_to_storage);
    }

    pub fn to_texture(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        mip_level: u32,
    ) {
        assert_eq!(src.size(), dst.size());
        self.run_internal(ctx, encoder, src, dst, mip_level, &self.storage_to_texture);
    }

    fn run_internal(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        mip_level: u32,
        pipeline: &wgpu::RenderPipeline,
    ) {
        assert_eq!(
            src.size(),
            dst.size(),
            "copy render expects to transfer pixels 1-by-1"
        );

        let src_view = src.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dst_view = dst.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Copy to Storage BindGroup"),
            layout: &self.texture_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&src_view),
            }],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Copy to Storage Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            ..wgpu::RenderPassDescriptor::default()
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &texture_bind_group, &[]);
        pass.draw(0..4, 0..1);
    }

    fn create_texture_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Copy Pipeline Texture Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            })
    }

    fn create_copy_pipelines(
        ctx: &GpuContext,
        texture_layout: &wgpu::BindGroupLayout,
        original_format: wgpu::TextureFormat,
    ) -> (wgpu::RenderPipeline, wgpu::RenderPipeline) {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MipMapper Copy Pipeline Layout"),
                bind_group_layouts: &[texture_layout],
                push_constant_ranges: &[],
            });

        let vs_module = crate::instruments::create_simple_shader_module_desc(
            Some("Quad Shader"),
            Self::SHADER_COPY_VERTEX,
        );
        let vs_module = ctx.device.create_shader_module(vs_module);

        let fs_module = crate::instruments::create_simple_shader_module_desc(
            Some("Copy Fragment Shader"),
            Self::SHADER_COPY_FRAGMENT,
        );
        let fs_module = ctx.device.create_shader_module(fs_module);

        let to_storage_pipeline =
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Copy to Storage Pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &vs_module,
                        entry_point: Some("vs_quad"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &fs_module,
                        entry_point: Some("fs_copy"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: FORMAT_STORAGE,
                            blend: None,
                            write_mask: wgpu::ColorWrites::default(),
                        })],
                    }),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });

        let to_texture_pipeline =
            ctx.device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Copy to Texture Pipeline"),
                    layout: Some(&layout),
                    vertex: wgpu::VertexState {
                        module: &vs_module,
                        entry_point: Some("vs_quad"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleStrip,
                        ..Default::default()
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &fs_module,
                        entry_point: Some("fs_copy"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: original_format,
                            blend: None,
                            write_mask: wgpu::ColorWrites::default(),
                        })],
                    }),
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState::default(),
                    multiview: None,
                    cache: None,
                });

        (to_storage_pipeline, to_texture_pipeline)
    }
}
