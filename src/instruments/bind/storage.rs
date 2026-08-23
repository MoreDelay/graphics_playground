//! Instruments concerning storage textures

use std::range::Range;

use iced::wgpu;

use crate::instruments::GpuContext;
use crate::instruments::bind::texture::SimpleTextureLayout;

/// The default storage texture color format
pub const FORMAT_STORAGE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// A texture that can be used as storage texture
pub struct SimpleStorageTexture(wgpu::Texture);

impl SimpleStorageTexture {
    /// Create an empty storage texture
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

    /// Copy the contents stored from this storage texture over to a normal texture
    pub fn copy_to_texture(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        machine: &StorageTextureCopyMachine,
        dst: &wgpu::Texture,
        mip_range: Range<u32>,
    ) {
        for mip_level in mip_range {
            machine.to_texture(ctx, encoder, self, dst, mip_level);
        }
    }

    /// Copy the contents from a normal texture into this storage texture
    pub fn copy_from_texture(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        machine: &StorageTextureCopyMachine,
        src: &wgpu::Texture,
        mip_range: Range<u32>,
    ) {
        for mip_level in mip_range {
            machine.to_storage(ctx, encoder, src, self, mip_level);
        }
    }

    /// Get the inner texture
    #[expect(dead_code)]
    pub const fn texture(&self) -> &wgpu::Texture {
        &self.0
    }
}

impl std::ops::Deref for SimpleStorageTexture {
    type Target = wgpu::Texture;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// A bind group layout for accessing two storage textures in a shader
pub struct StorageSrcDstLayout(wgpu::BindGroupLayout);

impl StorageSrcDstLayout {
    /// Create a new layout
    pub fn new(ctx: &GpuContext, label: Option<&str>) -> Self {
        let bind = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Rgba8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                ],
            });
        Self(bind)
    }
}

impl std::ops::Deref for StorageSrcDstLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Handles the copying between a normal and a storage texture
///
/// Because a storage texture can not use sRGB colors, and a direct copy of data does not perform
/// the any color space transformations, we need to use a full render pass to copy the color
/// correctly (while staying on the GPU). This struct contains all the necessary machinery to do
/// these simple renderings.
pub struct StorageTextureCopyMachine {
    /// The texture layout for the source
    texture_layout: SimpleTextureLayout,
    /// Pipeline to copy from texture to storage
    texture_to_storage: wgpu::RenderPipeline,
    /// Pipeline to copy from storage to texture
    storage_to_texture: wgpu::RenderPipeline,
}

impl StorageTextureCopyMachine {
    /// The copy machine vertex shader path
    const SHADER_COPY_VERTEX: &str = "package::image::quad";
    /// The copy machine fragment shader path
    const SHADER_COPY_FRAGMENT: &str = "package::mipmap::texture_copy";

    /// Create a new copy machine
    ///
    /// The original texture format limits the texture format that are compatible with this machine
    pub fn new(ctx: &GpuContext, original_format: wgpu::TextureFormat) -> Self {
        let texture_layout = SimpleTextureLayout::new(ctx, Some("Copy Pipeline Texture Layout"));
        let (texture_to_storage, storage_to_texture) =
            Self::create_copy_pipelines(ctx, &texture_layout, original_format);

        Self {
            texture_layout,
            texture_to_storage,
            storage_to_texture,
        }
    }

    /// Copy from a normal texture to a storage texture
    pub fn to_storage(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &SimpleStorageTexture,
        mip_level: u32,
    ) {
        assert_eq!(src.size(), dst.size(), "copy only when sizes equal");
        self.run_internal(ctx, encoder, src, dst, mip_level, &self.texture_to_storage);
    }

    /// Copy from a storage texture to a normal texture
    pub fn to_texture(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        src: &SimpleStorageTexture,
        dst: &wgpu::Texture,
        mip_level: u32,
    ) {
        assert_eq!(src.size(), dst.size(), "copy only when sizes equal");
        self.run_internal(ctx, encoder, src, dst, mip_level, &self.storage_to_texture);
    }

    /// Internal command that implements the copy rendering
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

    /// Create the internal pipelines of this machine
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
