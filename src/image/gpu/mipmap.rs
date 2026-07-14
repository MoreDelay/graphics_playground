use core::f32;
use std::num::NonZeroU64;

use iced::wgpu::util::DeviceExt as _;
use iced::wgpu::{self};
use image::EncodableLayout as _;

use crate::GpuContext;

pub struct MipMapper {
    compute_pipeline_downsampling: wgpu::ComputePipeline,
    compute_pipeline_convolution: wgpu::ComputePipeline,
    storage_texture_layout: wgpu::BindGroupLayout,
    #[expect(unused)]
    kernel_texture_layout: wgpu::BindGroupLayout,
}

impl MipMapper {
    const SHADER_DOWNSAMPLING: &str = "package::mipmapper";
    const SHADER_CONVOLUTION: &str = "package::gauss_filter";

    #[expect(clippy::too_many_lines)]
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

        let kernel_texture_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MipMapper Kernel Texture Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::ReadOnly,
                                format: wgpu::TextureFormat::R32Float,
                                view_dimension: wgpu::TextureViewDimension::D1,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout_downsampling =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("MipMapper Downsampling Pipeline Layout"),
                    bind_group_layouts: &[&storage_texture_layout],
                    push_constant_ranges: &[],
                });
        let compute_module = super::create_shader_module_desc(
            Some("Downsampling Shader"),
            Self::SHADER_DOWNSAMPLING,
        );
        let compute_module = ctx.device.create_shader_module(compute_module);
        let compute_pipeline_downsampling =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MipMapper Downsampling Pipeline"),
                    layout: Some(&pipeline_layout_downsampling),
                    module: &compute_module,
                    entry_point: Some("compute_mipmap"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        let pipeline_layout_convolution =
            ctx.device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("MipMapper Convolution Pipeline Layout"),
                    bind_group_layouts: &[&storage_texture_layout, &kernel_texture_layout],
                    push_constant_ranges: &[],
                });
        let compute_module =
            super::create_shader_module_desc(Some("Convolution Shader"), Self::SHADER_CONVOLUTION);
        let compute_module = ctx.device.create_shader_module(compute_module);
        let compute_pipeline_convolution =
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MipMapper Convolution Pipeline"),
                    layout: Some(&pipeline_layout_convolution),
                    module: &compute_module,
                    entry_point: Some("apply_kernel"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                });

        Self {
            compute_pipeline_downsampling,
            compute_pipeline_convolution,
            storage_texture_layout,
            kernel_texture_layout,
        }
    }

    #[expect(clippy::too_many_lines)]
    pub fn compute_mipmaps(&self, ctx: &GpuContext, texture: &wgpu::Texture) {
        use wgpu::TextureFormat::*;

        let sigma = 3.;

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

        let label = Some("MipMapper filtered-1d storage texture");
        let texture_filtered_1d = Self::create_storage_texture(ctx, label, texture);

        let label = Some("MipMapper filtered-2d storage texture");
        let texture_filtered_2d = Self::create_storage_texture(ctx, label, texture);

        let label = Some("MipMapper Complete storage texture");
        let texture_complete = Self::create_storage_texture(ctx, label, texture);
        {
            let src = wgpu::TexelCopyTextureInfo {
                mip_level: 0,
                ..texture.as_image_copy()
            };
            let dst = wgpu::TexelCopyTextureInfo {
                mip_level: 0,
                ..texture_complete.as_image_copy()
            };
            encoder.copy_texture_to_texture(src, dst, texture.size());
        }

        let kernel_layout = KernelBindGroupLayout::new(ctx);
        let kernel_bind = KernelBinding::new(sigma, ctx, &kernel_layout);

        let mut src_view = texture_complete.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        });

        let dispatch_x = texture_complete.width().div_ceil(16);
        let dispatch_y = texture_complete.height().div_ceil(16);
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MipMapper Compute Pass"),
                timestamp_writes: None,
            });
            for mip_level in 1..texture.mip_level_count() {
                pass.set_pipeline(&self.compute_pipeline_convolution);
                let dst_view = texture_filtered_1d.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mip_level - 1,
                    mip_level_count: Some(1),
                    ..Default::default()
                });
                let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MipMapper Filter-1d Bind Group"),
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
                kernel_bind.set_axis(ctx, Axis::X);
                pass.set_bind_group(0, &texture_bind_group, &[]);
                pass.set_bind_group(1, kernel_bind.bind_group(), &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                src_view = dst_view;

                let dst_view = texture_filtered_2d.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mip_level - 1,
                    mip_level_count: Some(1),
                    ..Default::default()
                });
                let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MipMapper Filter-2d Bind Group"),
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
                kernel_bind.set_axis(ctx, Axis::Y);
                pass.set_bind_group(0, &texture_bind_group, &[]);
                // kept from last call
                // pass.set_bind_group(1, kernel_bind.bind_group(), &[]);
                pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
                src_view = dst_view;

                pass.set_pipeline(&self.compute_pipeline_downsampling);
                let dst_view = texture_complete.create_view(&wgpu::TextureViewDescriptor {
                    base_mip_level: mip_level,
                    mip_level_count: Some(1),
                    ..Default::default()
                });
                let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("MipMapper Downsampling BindGroup"),
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
        println!("complete, now copy over");

        // copy computed mip maps over to texture
        {
            let mut size = texture.size();
            for mip_level in 1..texture.mip_level_count() {
                size.width /= 2;
                size.height /= 2;

                let src = wgpu::TexelCopyTextureInfo {
                    mip_level,
                    ..texture_complete.as_image_copy()
                };
                let dst = wgpu::TexelCopyTextureInfo {
                    mip_level,
                    ..texture.as_image_copy()
                };
                encoder.copy_texture_to_texture(src, dst, size);
            }
        }
        println!("submit now");
        ctx.queue.submit([encoder.finish()]);
        println!("done");
    }

    fn create_storage_texture(
        ctx: &GpuContext,
        label: Option<&str>,
        base: &wgpu::Texture,
    ) -> wgpu::Texture {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: base.size(),
            mip_level_count: base.mip_level_count(),
            sample_count: base.sample_count(),
            dimension: base.dimension(),
            format: base.format().remove_srgb_suffix(),
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }
}

struct KernelBinding {
    bind_group: wgpu::BindGroup,
    #[expect(unused)]
    storage_texture: wgpu::Texture,
    buffer: wgpu::Buffer,
    kernel_size: u32,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X = 0,
    Y = 1,
}

impl KernelBinding {
    fn new(sigma: f32, ctx: &GpuContext, layout: &KernelBindGroupLayout) -> Self {
        let kernel = Self::create_gauss_kernel(sigma);
        let storage_texture = Self::create_kernel_texture(ctx, &kernel);
        let view = storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

        #[expect(clippy::cast_possible_truncation)]
        let kernel_size = kernel.len() as u32;
        let data = KernelInfoRaw {
            axis: 0,
            offset: kernel_size / 2,
        };
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("KernelInfo Buffer"),
                contents: bytemuck::cast_slice(&[data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let entries = &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffer.as_entire_binding(),
            },
        ];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("KernelInfo Bind Group"),
            layout,
            entries,
        });

        Self {
            bind_group,
            storage_texture,
            buffer,
            kernel_size,
        }
    }

    fn set_axis(&self, ctx: &GpuContext, axis: Axis) {
        const SIZE: NonZeroU64 =
            NonZeroU64::new(std::mem::size_of::<KernelInfoRaw>() as u64).expect("struct not empty");

        let data = KernelInfoRaw {
            axis: axis as u32,
            offset: self.kernel_size / 2,
        };

        let mut view = ctx
            .queue
            .write_buffer_with(&self.buffer, 0, SIZE)
            .expect("failed creating temporary buffer for upload");

        view.copy_from_slice(bytemuck::cast_slice(&[data]));
    }

    const fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }

    fn create_gauss_kernel(sigma: f32) -> Vec<f32> {
        assert!(sigma > 0., "standard deviation must be non-negative");
        #[expect(clippy::cast_possible_truncation)]
        let radius = (3. * sigma).ceil() as usize;
        let mut kernel = vec![0.; radius + 1 + radius];
        for (i, k) in kernel.iter_mut().enumerate() {
            #[expect(clippy::cast_precision_loss)]
            let t = i as f32 - radius as f32;
            let factor = 1. / ((2. * f32::consts::PI).sqrt() * sigma);
            let exponent = (-t * t) / (2. * sigma * sigma);
            let gauss_value = factor * exponent.exp();

            *k = gauss_value;
        }
        kernel
    }

    fn create_kernel_texture(ctx: &GpuContext, kernel: &[f32]) -> wgpu::Texture {
        let n_kernel = kernel.len();
        let kernel = kernel.as_bytes();

        #[expect(clippy::cast_possible_truncation)]
        let width = n_kernel as u32;
        let size = wgpu::Extent3d {
            width,
            height: 1,
            depth_or_array_layers: 1,
        };
        let format = wgpu::TextureFormat::R32Float;
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Gauss Kernel Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D1,
            format,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let texel_bytes = format.block_copy_size(None).expect("should be 4");
        let bytes_per_row = texel_bytes * size.width;
        assert_eq!(
            kernel.len(),
            bytes_per_row as usize,
            "Bytes written should correspond to bytes we have"
        );
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            kernel,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: Some(1),
            },
            size,
        );

        texture
    }
}

pub struct KernelBindGroupLayout(wgpu::BindGroupLayout);

impl KernelBindGroupLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Kernel Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::ReadOnly,
                            format: wgpu::TextureFormat::R32Float,
                            view_dimension: wgpu::TextureViewDimension::D1,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        Self(layout)
    }
}

impl std::ops::Deref for KernelBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct KernelInfoRaw {
    /// Axis to apply the kernel on (x for 0, y for 1)
    pub axis: u32,
    /// How much the kernel is offset from the target location
    ///
    /// In the (1-dimensional) formula `SUM_i [K(i) * T(p - o + i)]`, where p is the target
    /// location, K is the kernel array and T is the texture array, corresponds to the offset
    /// o.
    pub offset: u32,
}
