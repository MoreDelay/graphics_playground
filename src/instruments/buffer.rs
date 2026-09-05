//! Instruments for uniform buffers

use std::marker::PhantomData;
use std::num::NonZeroU64;

use iced::wgpu;
use iced::wgpu::util::DeviceExt as _;

use crate::instruments::GpuContext;

/// A marker trait for types that can be used as raw data provides for uniform buffers
pub trait BufferRaw: Copy + Clone + bytemuck::Pod + bytemuck::Zeroable {}
impl<T> BufferRaw for T where T: Copy + Clone + bytemuck::Pod + bytemuck::Zeroable {}

/// A simple uniform buffer
pub struct SimpleBuffer<R: BufferRaw> {
    /// The gpu buffer
    buffer: wgpu::Buffer,
    /// Marker to associate this buffer with some data provider type
    marker: PhantomData<R>,
}

impl<R: BufferRaw> SimpleBuffer<R> {
    /// The number of bytes stored in this buffer
    const SIZE: NonZeroU64 =
        NonZeroU64::new(std::mem::size_of::<R>() as u64).expect("struct not empty");

    /// Create a new buffer
    pub fn new(ctx: &GpuContext, init: R, label: Option<&str>) -> Self {
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: bytemuck::cast_slice(&[init]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let marker = PhantomData;
        Self { buffer, marker }
    }

    /// Update the contents of this buffer
    pub fn update(&self, ctx: &GpuContext, data: R) {
        ctx.queue
            .write_buffer_with(&self.buffer, 0, Self::SIZE)
            .expect("failed creating temporary buffer for upload")
            .copy_from_slice(bytemuck::cast_slice(&[data]));
    }

    /// Get this buffer as a [`wgpu::BindingResource`]
    pub fn resource(&self) -> wgpu::BindingResource<'_> {
        self.buffer.as_entire_binding()
    }
}

/// A bind group for a uniform buffer
pub struct SimpleBufferBind<R: BufferRaw> {
    /// The buffer used in the binding
    buffer: SimpleBuffer<R>,
    /// The bind group
    bind: wgpu::BindGroup,
}

impl<R: BufferRaw> SimpleBufferBind<R> {
    /// Create a new binding
    pub fn new(
        ctx: &GpuContext,
        buffer: SimpleBuffer<R>,
        layout: &SimpleBufferBindLayout,
        label: Option<&str>,
    ) -> Self {
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout: &layout.0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.resource(),
            }],
        });
        Self { buffer, bind }
    }

    /// Access the buffer of this binding
    pub const fn buffer(&self) -> &SimpleBuffer<R> {
        &self.buffer
    }
}

impl<R: BufferRaw> std::ops::Deref for SimpleBufferBind<R> {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.bind
    }
}

/// The layout for [`SimpleBufferBind`]
pub struct SimpleBufferBindLayout(wgpu::BindGroupLayout);

impl SimpleBufferBindLayout {
    /// Create a new layout
    pub fn new(ctx: &GpuContext, label: Option<&str>) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        Self(layout)
    }
}

impl std::ops::Deref for SimpleBufferBindLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
