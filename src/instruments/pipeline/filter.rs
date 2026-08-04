use iced::wgpu;
use image::EncodableLayout as _;

use crate::instruments::GpuContext;
use crate::instruments::bind::storage::SimpleStorageTexture;
use crate::instruments::buffer::SimpleBuffer;

pub struct ConvolutionPipeline(wgpu::ComputePipeline);

impl ConvolutionPipeline {
    const SHADER_CONVOLUTION: &str = "package::mipmap::convolution";

    pub fn new(ctx: &GpuContext, layout: &ConvolutionPipelineLayout, label: Option<&str>) -> Self {
        let module = crate::instruments::create_simple_shader_module_desc(
            Some("Convolution Shader"),
            Self::SHADER_CONVOLUTION,
        );
        let module = ctx.device.create_shader_module(module);
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label,
                layout: Some(layout),
                module: &module,
                entry_point: Some("convolve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self(pipeline)
    }

    #[expect(clippy::too_many_arguments)]
    pub fn run(
        &self,
        ctx: &GpuContext,
        pass: &mut wgpu::ComputePass,
        storage_layout: &StorageSrcDstLayout,
        storage_src: &SimpleStorageTexture,
        storage_scratch: &SimpleStorageTexture,
        storage_dst: &SimpleStorageTexture,
        kernel_bind: &KernelBinding,
        mip_level: u32,
    ) {
        assert!(
            !std::ptr::eq(storage_src, storage_scratch),
            "source and scratch storage texture can not be the same",
        );
        assert!(
            !std::ptr::eq(storage_scratch, storage_dst),
            "destination and scratch storage texture can not be the same",
        );

        let runner = ConvolutionRunner {
            storage_layout,
            pipeline: self,
            storage_src,
            storage_scratch,
            storage_dst,
            kernel_bind,
        };

        runner.run(ctx, pass, mip_level);
    }
}

struct ConvolutionRunner<'a> {
    storage_layout: &'a StorageSrcDstLayout,
    pipeline: &'a ConvolutionPipeline,
    storage_src: &'a SimpleStorageTexture,
    storage_scratch: &'a SimpleStorageTexture,
    storage_dst: &'a SimpleStorageTexture,
    kernel_bind: &'a KernelBinding,
}

impl ConvolutionRunner<'_> {
    fn run(&self, ctx: &GpuContext, pass: &mut wgpu::ComputePass, mip_level: u32) {
        self.run_filter_src_to_scratch(ctx, pass, Axis::Y, mip_level);
        self.run_filter_scratch_to_dst(ctx, pass, Axis::X, mip_level);
    }

    fn run_filter_src_to_scratch(
        &self,
        ctx: &GpuContext,
        pass: &mut wgpu::ComputePass,
        axis: Axis,
        mip_level: u32,
    ) {
        let src_view = self.storage_src.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dst_view = self
            .storage_scratch
            .create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip_level,
                mip_level_count: Some(1),
                ..Default::default()
            });
        let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MipMapper Filter-1d Bind Group"),
            layout: self.storage_layout,
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
        let kernel_bind = match axis {
            Axis::X => self.kernel_bind.bind_group_x(),
            Axis::Y => self.kernel_bind.bind_group_y(),
        };

        // divide by 2^mip_level
        let dispatch_x = self.storage_src.width() >> mip_level;
        let dispatch_y = self.storage_src.height() >> mip_level;
        let dispatch_x = dispatch_x.div_ceil(16);
        let dispatch_y = dispatch_y.div_ceil(16);

        pass.set_pipeline(&self.pipeline.0);
        pass.set_bind_group(0, &texture_bind_group, &[]);
        pass.set_bind_group(1, kernel_bind, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }

    fn run_filter_scratch_to_dst(
        &self,
        ctx: &GpuContext,
        pass: &mut wgpu::ComputePass,
        axis: Axis,
        mip_level: u32,
    ) {
        let src_view = self
            .storage_scratch
            .create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: mip_level,
                mip_level_count: Some(1),
                ..Default::default()
            });
        let dst_view = self.storage_dst.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MipMapper Filter-1d Bind Group"),
            layout: self.storage_layout,
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
        let kernel_bind = match axis {
            Axis::X => self.kernel_bind.bind_group_x(),
            Axis::Y => self.kernel_bind.bind_group_y(),
        };

        // divide by 2^mip_level
        let dispatch_x = self.storage_dst.width() >> mip_level;
        let dispatch_y = self.storage_dst.height() >> mip_level;
        let dispatch_x = dispatch_x.div_ceil(16);
        let dispatch_y = dispatch_y.div_ceil(16);

        pass.set_pipeline(&self.pipeline.0);
        pass.set_bind_group(0, &texture_bind_group, &[]);
        pass.set_bind_group(1, kernel_bind, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}

pub struct ConvolutionPipelineLayout(wgpu::PipelineLayout);

impl ConvolutionPipelineLayout {
    pub fn new(
        ctx: &GpuContext,
        storage_layout: &StorageSrcDstLayout,
        kernel_layout: &KernelLayout,
        label: Option<&str>,
    ) -> Self {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label,
                bind_group_layouts: &[storage_layout, kernel_layout],
                push_constant_ranges: &[],
            });
        Self(layout)
    }
}

impl std::ops::Deref for ConvolutionPipelineLayout {
    type Target = wgpu::PipelineLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct StorageSrcDstLayout(wgpu::BindGroupLayout);

impl StorageSrcDstLayout {
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

pub struct KernelLayout(wgpu::BindGroupLayout);

impl KernelLayout {
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
        Self(bind)
    }
}

impl std::ops::Deref for KernelLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X = 0,
    Y = 1,
}

pub struct KernelBinding {
    bind_group_x: wgpu::BindGroup,
    bind_group_y: wgpu::BindGroup,
    #[expect(unused)]
    storage_texture: wgpu::Texture,
    #[expect(unused)]
    buffer_x: SimpleBuffer<KernelInfoRaw>,
    #[expect(unused)]
    buffer_y: SimpleBuffer<KernelInfoRaw>,
    #[expect(unused)]
    kernel_size: u32,
}

impl KernelBinding {
    pub fn new(
        ctx: &GpuContext,
        layout: &KernelLayout,
        kernel: &[f32],
        label: Option<&str>,
    ) -> Self {
        let storage_texture = Self::create_kernel_texture(ctx, kernel);
        let view = storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

        #[expect(clippy::cast_possible_truncation)]
        let kernel_size = kernel.len() as u32;
        let data = KernelInfoRaw {
            axis: 0,
            offset: kernel_size / 2,
        };
        let buffer_x = SimpleBuffer::new(ctx, data, Some("KernelInfo Buffer X"));
        let data = KernelInfoRaw {
            axis: 1,
            offset: kernel_size / 2,
        };
        let buffer_y = SimpleBuffer::new(ctx, data, Some("KernelInfo Buffer Y"));
        let bind_group_x = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_x.resource(),
                },
            ],
        });
        let bind_group_y = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buffer_y.resource(),
                },
            ],
        });

        Self {
            bind_group_x,
            bind_group_y,
            storage_texture,
            buffer_x,
            buffer_y,
            kernel_size,
        }
    }

    const fn bind_group_x(&self) -> &wgpu::BindGroup {
        &self.bind_group_x
    }

    const fn bind_group_y(&self) -> &wgpu::BindGroup {
        &self.bind_group_y
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

impl std::ops::Deref for KernelBinding {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.bind_group_x
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
