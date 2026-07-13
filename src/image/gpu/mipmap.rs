use iced::wgpu;

use super::SHADER_ROOT;
use crate::GpuContext;

pub struct MipMapper {
    compute_pipeline: wgpu::ComputePipeline,
    storage_texture_layout: wgpu::BindGroupLayout,
}

impl MipMapper {
    const SHADER: &str = "package::mipmapper";

    pub fn new(ctx: &GpuContext) -> Self {
        let storage_texture_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MipMapper Storage Texture Layout"),
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

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MipMapper Pipeline Layout"),
                bind_group_layouts: &[&storage_texture_layout],
                push_constant_ranges: &[],
            });
        let compute_module = &Self::SHADER.parse().expect("module path invalid");
        let compute_module = wesl::Wesl::new(SHADER_ROOT)
            .compile(compute_module)
            .inspect_err(|e| eprintln!("WESL error: {e}"))
            .expect("shader invalid")
            .to_string();
        let compute_module = wgpu::ShaderModuleDescriptor {
            label: Some("ColorNormalShader"),
            source: wgpu::ShaderSource::Wgsl(compute_module.into()),
        };
        let compute_module = ctx.device.create_shader_module(compute_module);
        let compute_pipeline =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MipMapper Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &compute_module,
                    entry_point: Some("compute_mipmap"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        Self {
            compute_pipeline,
            storage_texture_layout,
        }
    }

    #[expect(clippy::too_many_lines)]
    pub fn compute_mipmaps(&self, ctx: &GpuContext, texture: &wgpu::Texture) {
        use wgpu::TextureFormat::*;

        assert!(
            matches!(texture.format(), Rgba8Unorm | Rgba8UnormSrgb),
            "unexpected texture format"
        );

        if texture.mip_level_count() == 1 {
            return;
        }

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        let (mut src_view, maybe_tmp) = if texture
            .usage()
            .contains(wgpu::TextureUsages::STORAGE_BINDING)
        {
            (
                texture.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    ..Default::default()
                }),
                None,
            )
        } else {
            let tmp = ctx.device.create_texture(&wgpu::TextureDescriptor {
                label: Some("MipMapper temporary storage texture"),
                size: texture.size(),
                mip_level_count: texture.mip_level_count(),
                sample_count: texture.sample_count(),
                dimension: texture.dimension(),
                format: texture.format().remove_srgb_suffix(),
                usage: wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::COPY_DST
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            encoder.copy_texture_to_texture(
                texture.as_image_copy(),
                tmp.as_image_copy(),
                tmp.size(),
            );
            (
                tmp.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: 0,
                    mip_level_count: Some(1),
                    ..Default::default()
                }),
                Some(tmp),
            )
        };

        let dispatch_x = texture.width().div_ceil(16);
        let dispatch_y = texture.height().div_ceil(16);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MipMapper Compute Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.compute_pipeline);
            for mip_level in 1..texture.mip_level_count() {
                let dst_view = src_view
                    .texture()
                    .create_view(&wgpu::TextureViewDescriptor {
                        base_mip_level: mip_level,
                        mip_level_count: Some(1),
                        ..Default::default()
                    });
                let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MipMapper BindGroup"),
                    layout: &self.storage_texture_layout,
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
                pass.set_bind_group(0, &texture_bind_group, &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);

                src_view = dst_view;
            }
        }

        // copy computed mip maps over to texture
        if let Some(tmp) = maybe_tmp {
            let mut size = tmp.size();
            for mip_level in 0..tmp.mip_level_count() {
                let src = wgpu::TexelCopyTextureInfo {
                    mip_level,
                    ..tmp.as_image_copy()
                };
                let dst = wgpu::TexelCopyTextureInfo {
                    mip_level,
                    ..texture.as_image_copy()
                };
                encoder.copy_texture_to_texture(src, dst, size);

                size.width /= 2;
                size.height /= 2;
            }
        }
        ctx.queue.submit([encoder.finish()]);
    }
}
