//! Instruments to run filter convolutions

use iced::wgpu;
use image::EncodableLayout as _;

use crate::instruments::GpuContext;
use crate::instruments::bind::storage::{SimpleStorageTexture, StorageSrcDstLayout};
use crate::instruments::buffer::SimpleBuffer;

/// The convolution pipeline
pub struct ConvolutionPipeline(wgpu::ComputePipeline);

impl ConvolutionPipeline {
    /// The convolution compute shader path
    const SHADER_CONVOLUTION: &str = "package::mipmap::convolution";

    /// Create a new convolution pipeline
    pub fn new(
        ctx: &GpuContext,
        layout: &ConvolutionPipelineLayout,
        label_shader: Option<&str>,
        label_pipeline: Option<&str>,
    ) -> Self {
        let module = crate::instruments::create_simple_shader_module_desc(
            label_shader,
            Self::SHADER_CONVOLUTION,
        );
        let module = ctx.device.create_shader_module(module);
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: label_pipeline,
                layout: Some(layout),
                module: &module,
                entry_point: Some("convolve"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        Self(pipeline)
    }

    /// Execute a convolution
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

        self.run_internal(
            ctx,
            pass,
            storage_layout,
            storage_src,
            storage_scratch,
            kernel_bind,
            mip_level,
            Axis::X,
        );

        self.run_internal(
            ctx,
            pass,
            storage_layout,
            storage_scratch,
            storage_dst,
            kernel_bind,
            mip_level,
            Axis::Y,
        );
    }

    /// Run the convolution on a specific axis
    #[expect(clippy::too_many_arguments)]
    fn run_internal(
        &self,
        ctx: &GpuContext,
        pass: &mut wgpu::ComputePass,
        storage_layout: &StorageSrcDstLayout,
        storage_src: &SimpleStorageTexture,
        storage_dst: &SimpleStorageTexture,
        kernel_bind: &KernelBinding,
        mip_level: u32,
        axis: Axis,
    ) {
        let src_view = storage_src.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dst_view = storage_dst.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MipMapper Filter-1d Bind Group"),
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
        let kernel_bind = match axis {
            Axis::X => kernel_bind.bind_group_x(),
            Axis::Y => kernel_bind.bind_group_y(),
        };

        // divide by 2^mip_level
        let dispatch_x = storage_src.width() >> mip_level;
        let dispatch_y = storage_src.height() >> mip_level;
        let dispatch_x = dispatch_x.div_ceil(16);
        let dispatch_y = dispatch_y.div_ceil(16);

        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, &texture_bind_group, &[]);
        pass.set_bind_group(1, kernel_bind, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}

/// The layout for [`ConvolutionPipeline`]
pub struct ConvolutionPipelineLayout(wgpu::PipelineLayout);

impl ConvolutionPipelineLayout {
    /// Create a new layout
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

/// The bindings for executing a 2d convolution
pub struct KernelBinding {
    /// The metadata for the run on the X-axis
    bind_group_x: wgpu::BindGroup,
    /// The metadata for the run on the Y-axis
    bind_group_y: wgpu::BindGroup,

    /// The storage texture holding all kernel weight values
    #[expect(unused, reason = "used in bindings above")]
    storage_texture: wgpu::Texture,
    /// Kernel metadata for X-axis
    #[expect(unused, reason = "used in x-binding above")]
    buffer_x: SimpleBuffer<KernelInfoRaw>,
    /// Kernel metadata for Y-axis
    #[expect(unused, reason = "used in y-binding above")]
    buffer_y: SimpleBuffer<KernelInfoRaw>,
    /// The number of weights used in this kernel
    #[expect(unused)]
    kernel_size: u32,
}

impl KernelBinding {
    /// Create a new kernel weights binding
    pub fn new(
        ctx: &GpuContext,
        layout: &KernelLayout,
        kernel: &[f32],
        label: Option<&str>,
    ) -> Self {
        let storage_texture = Self::create_kernel_texture(ctx, kernel);
        let view = storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

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

    /// Get the bind group for the X-axis pass
    const fn bind_group_x(&self) -> &wgpu::BindGroup {
        &self.bind_group_x
    }

    /// Get the bind group for the Y-axis pass
    const fn bind_group_y(&self) -> &wgpu::BindGroup {
        &self.bind_group_y
    }

    /// Create the storage texture holding the kernel weights
    fn create_kernel_texture(ctx: &GpuContext, kernel: &[f32]) -> wgpu::Texture {
        let n_kernel = kernel.len();
        let kernel = kernel.as_bytes();

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

/// The layout for [`KernelBinding`]
pub struct KernelLayout(wgpu::BindGroupLayout);

impl KernelLayout {
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

/// The axis on which a convolution pass is executed on
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    X = 0,
    Y = 1,
}

/// Raw convolution metadata used in shader
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct KernelInfoRaw {
    /// Axis to apply the kernel on (x for 0, y for 1)
    axis: u32,
    #[expect(clippy::doc_markdown)]
    /// How much the kernel is offset from the target location
    ///
    /// In the (1-dimensional) formula $\sum_i [K(i) \cdot T(p - o + i)],$ where p is the target
    /// location, K is the kernel array and T is the texture array, corresponds to the offset
    /// o.
    offset: u32,
}
