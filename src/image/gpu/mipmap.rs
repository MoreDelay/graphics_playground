use core::f32;
use std::num::NonZeroU64;

use iced::wgpu::util::DeviceExt as _;
use iced::wgpu::{self};
use image::EncodableLayout as _;

use crate::GpuContext;
use crate::image::filters::GaussFilter;

pub struct MipMapper {
    pipeline_downsampling: wgpu::ComputePipeline,
    pipeline_convolution: wgpu::ComputePipeline,
    storage_layout: wgpu::BindGroupLayout,
    #[expect(unused)]
    kernel_layout: wgpu::BindGroupLayout,
}

impl MipMapper {
    const SHADER_HALFING: &str = "package::mipmap::halfing";
    const SHADER_CONVOLUTION: &str = "package::mipmap::convolution";

    pub fn new(ctx: &GpuContext) -> Self {
        let storage_layout = Self::create_texture_storage_layout(ctx);
        let kernel_layout = Self::create_kernel_layout(ctx);

        let pipeline_downsampling = Self::create_pipeline_downsampling(ctx, &storage_layout);
        let pipeline_convolution =
            Self::create_pipeline_convolution(ctx, &storage_layout, &kernel_layout);

        Self {
            pipeline_downsampling,
            pipeline_convolution,
            storage_layout,
            kernel_layout,
        }
    }

    pub fn compute_mipmaps(&self, ctx: &GpuContext, texture: &wgpu::Texture) {
        assert!(
            texture.format().is_srgb(),
            "expect sRGB textures (due to copy pipeline)"
        );

        let Some(runner) = MipMapRunner::new(ctx, self, texture) else {
            return;
        };
        runner.run(ctx);
    }

    fn create_texture_storage_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
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
            })
    }

    fn create_kernel_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
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
            })
    }

    fn create_pipeline_downsampling(
        ctx: &GpuContext,
        storage_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MipMapper Downsampling Pipeline Layout"),
                bind_group_layouts: &[storage_layout],
                push_constant_ranges: &[],
            });
        let module = crate::gpu::create_simple_shader_module_desc(
            Some("Downsampling Shader"),
            Self::SHADER_HALFING,
        );
        let module = ctx.device.create_shader_module(module);
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MipMapper Downsampling Pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("halfing"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }

    fn create_pipeline_convolution(
        ctx: &GpuContext,
        storage_layout: &wgpu::BindGroupLayout,
        kernel_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MipMapper Convolution Pipeline Layout"),
                bind_group_layouts: &[storage_layout, kernel_layout],
                push_constant_ranges: &[],
            });
        let module = crate::gpu::create_simple_shader_module_desc(
            Some("Convolution Shader"),
            Self::SHADER_CONVOLUTION,
        );
        let module = ctx.device.create_shader_module(module);
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MipMapper Convolution Pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("convolve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
    }
}

struct MipMapRunner<'a> {
    mip_mapper: &'a MipMapper,
    texture: &'a wgpu::Texture,
    copy_helper: StorageTextureCopyHelper,
    texture_filtered_1d: wgpu::Texture,
    texture_filtered_2d: wgpu::Texture,
    texture_downsampled: wgpu::Texture,
    kernel_bind: KernelBinding,
}

impl<'a> MipMapRunner<'a> {
    const FORMAT_STORAGE: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

    fn new(
        ctx: &GpuContext,
        mip_mapper: &'a MipMapper,
        texture: &'a wgpu::Texture,
    ) -> Option<Self> {
        use wgpu::TextureFormat::*;

        const SIGMA: f32 = 0.5;

        assert!(
            texture.format() == Rgba8UnormSrgb,
            "only handling this format atm due to copy"
        );

        if texture.mip_level_count() == 1 {
            return None;
        }

        let label = Some("MipMapper filtered-1d storage texture");
        let texture_filtered_1d = Self::create_storage_texture(ctx, texture, label);

        let label = Some("MipMapper filtered-2d storage texture");
        let texture_filtered_2d = Self::create_storage_texture(ctx, texture, label);

        let label = Some("MipMapper downsampled storage texture");
        let texture_downsampled = Self::create_storage_texture(ctx, texture, label);

        let kernel_layout = KernelBindGroupLayout::new(ctx);
        let kernel_bind = KernelBinding::new(SIGMA, ctx, &kernel_layout);

        let copy_helper = StorageTextureCopyHelper::new(ctx, texture.format());

        Some(Self {
            mip_mapper,
            texture,
            copy_helper,
            texture_filtered_1d,
            texture_filtered_2d,
            texture_downsampled,
            kernel_bind,
        })
    }

    fn run(self, ctx: &GpuContext) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        // copy over start texture to base level in downsampled texture stack
        self.copy_helper.to_storage(
            ctx,
            &mut encoder,
            self.texture,
            &self.texture_downsampled,
            0,
        );

        // run mip map construction
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MipMapper Compute Pass"),
                timestamp_writes: None,
            });

            let mip_level_iterations = self.texture.mip_level_count() - 1;
            for mip_level in 0..mip_level_iterations {
                self.run_filter_over_x(ctx, &mut pass, mip_level);
                self.run_filter_over_y(ctx, &mut pass, mip_level);
                self.run_downsampling(ctx, &mut pass, mip_level);
            }
        }

        // copy computed mip maps over to texture
        for mip_level in 1..self.texture.mip_level_count() {
            self.copy_helper.to_texture(
                ctx,
                &mut encoder,
                &self.texture_downsampled,
                self.texture,
                mip_level,
            );
        }

        ctx.queue.submit([encoder.finish()]);

        // store_texture_as_image(ctx, self.texture, std::path::Path::new("debug.png"));
    }

    fn run_filter_over_x(&self, ctx: &GpuContext, pass: &mut wgpu::ComputePass, mip_level: u32) {
        let src_view = self
            .texture_downsampled
            .create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip_level,
                mip_level_count: Some(1),
                ..Default::default()
            });
        let dst_view = self
            .texture_filtered_1d
            .create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip_level,
                mip_level_count: Some(1),
                ..Default::default()
            });
        let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MipMapper Filter-1d Bind Group"),
            layout: &self.mip_mapper.storage_layout,
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
        self.kernel_bind.set_axis(ctx, Axis::X);

        let dispatch_x = self.texture_downsampled.width() >> mip_level; // divide by 2^mip_level
        let dispatch_y = self.texture_downsampled.height() >> mip_level;
        let dispatch_x = dispatch_x.div_ceil(16);
        let dispatch_y = dispatch_y.div_ceil(16);

        pass.set_pipeline(&self.mip_mapper.pipeline_convolution);
        pass.set_bind_group(0, &texture_bind_group, &[]);
        pass.set_bind_group(1, self.kernel_bind.bind_group(), &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    fn run_filter_over_y(&self, ctx: &GpuContext, pass: &mut wgpu::ComputePass, mip_level: u32) {
        let src_view = self
            .texture_filtered_1d
            .create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip_level,
                mip_level_count: Some(1),
                ..Default::default()
            });
        let dst_view = self
            .texture_filtered_2d
            .create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip_level,
                mip_level_count: Some(1),
                ..Default::default()
            });
        let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MipMapper Filter-1d Bind Group"),
            layout: &self.mip_mapper.storage_layout,
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
        self.kernel_bind.set_axis(ctx, Axis::Y);

        let dispatch_x = self.texture_downsampled.width() >> mip_level; // divide by 2^mip_level
        let dispatch_y = self.texture_downsampled.height() >> mip_level;
        let dispatch_x = dispatch_x.div_ceil(16);
        let dispatch_y = dispatch_y.div_ceil(16);

        pass.set_pipeline(&self.mip_mapper.pipeline_convolution);
        pass.set_bind_group(0, &texture_bind_group, &[]);
        pass.set_bind_group(1, self.kernel_bind.bind_group(), &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    fn run_downsampling(
        &self,
        ctx: &GpuContext,
        pass: &mut wgpu::ComputePass,
        source_mip_level: u32,
    ) {
        let target_mip_level = source_mip_level + 1;

        let src_view = self
            .texture_filtered_2d
            .create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: source_mip_level,
                mip_level_count: Some(1),
                ..Default::default()
            });
        let dst_view = self
            .texture_downsampled
            .create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: target_mip_level,
                mip_level_count: Some(1),
                ..Default::default()
            });
        let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MipMapper Downsampling BindGroup"),
            layout: &self.mip_mapper.storage_layout,
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

        let dispatch_x = self.texture_downsampled.width() >> target_mip_level; // divide by 2^mip_level
        let dispatch_y = self.texture_downsampled.height() >> target_mip_level;
        let dispatch_x = dispatch_x.div_ceil(16);
        let dispatch_y = dispatch_y.div_ceil(16);

        pass.set_pipeline(&self.mip_mapper.pipeline_downsampling);
        pass.set_bind_group(0, &texture_bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    fn create_storage_texture(
        ctx: &GpuContext,
        base: &wgpu::Texture,
        label: Option<&str>,
    ) -> wgpu::Texture {
        ctx.device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: base.size(),
            mip_level_count: base.mip_level_count(),
            sample_count: base.sample_count(),
            dimension: base.dimension(),
            format: Self::FORMAT_STORAGE,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
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
        let gauss_filter = GaussFilter::new(sigma).expect("sigma should be non-negative");
        let kernel = gauss_filter.blur_kernel();
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

struct KernelBindGroupLayout(wgpu::BindGroupLayout);

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
struct KernelInfoRaw {
    /// Axis to apply the kernel on (x for 0, y for 1)
    axis: u32,
    /// How much the kernel is offset from the target location
    ///
    /// In the (1-dimensional) formula `SUM_i [K(i) * T(p - o + i)]`, where p is the target
    /// location, K is the kernel array and T is the texture array, corresponds to the offset
    /// o.
    offset: u32,
}

struct StorageTextureCopyHelper {
    texture_layout: wgpu::BindGroupLayout,
    texture_to_storage: wgpu::RenderPipeline,
    storage_to_texture: wgpu::RenderPipeline,
}

impl StorageTextureCopyHelper {
    const SHADER_COPY_VERTEX: &str = "package::image::quad";
    const SHADER_COPY_FRAGMENT: &str = "package::mipmap::texture_copy";

    fn new(ctx: &GpuContext, format: wgpu::TextureFormat) -> Self {
        let texture_layout = Self::create_texture_layout(ctx);
        let (texture_to_storage, storage_to_texture) =
            Self::create_copy_pipelines(ctx, &texture_layout, format);

        Self {
            texture_layout,
            texture_to_storage,
            storage_to_texture,
        }
    }

    fn to_storage(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        mip_level: u32,
    ) {
        self.run_internal(ctx, encoder, src, dst, mip_level, &self.texture_to_storage);
    }

    fn to_texture(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        src: &wgpu::Texture,
        dst: &wgpu::Texture,
        mip_level: u32,
    ) {
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

        let vs_module = crate::gpu::create_simple_shader_module_desc(
            Some("Quad Shader"),
            Self::SHADER_COPY_VERTEX,
        );
        let vs_module = ctx.device.create_shader_module(vs_module);

        let fs_module = crate::gpu::create_simple_shader_module_desc(
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
                            format: MipMapRunner::FORMAT_STORAGE,
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
