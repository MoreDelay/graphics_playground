use std::marker::PhantomData;
use std::num::NonZeroU64;

use iced::wgpu;
use iced::wgpu::util::DeviceExt as _;

use crate::instruments::GpuContext;

pub trait BufferRaw: Copy + Clone + bytemuck::Pod + bytemuck::Zeroable {}
impl<T> BufferRaw for T where T: Copy + Clone + bytemuck::Pod + bytemuck::Zeroable {}

pub struct SimpleBuffer<R: BufferRaw> {
    buffer: wgpu::Buffer,
    marker: PhantomData<R>,
}

impl<R: BufferRaw> SimpleBuffer<R> {
    const SIZE: NonZeroU64 =
        NonZeroU64::new(std::mem::size_of::<R>() as u64).expect("struct not empty");

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

    pub fn update(&self, ctx: &GpuContext, data: R) {
        ctx.queue
            .write_buffer_with(&self.buffer, 0, Self::SIZE)
            .expect("failed creating temporary buffer for upload")
            .copy_from_slice(bytemuck::cast_slice(&[data]));
    }

    pub fn resource(&self) -> wgpu::BindingResource<'_> {
        self.buffer.as_entire_binding()
    }
}

pub struct SimpleBufferBind<R: BufferRaw> {
    buffer: SimpleBuffer<R>,
    bind: wgpu::BindGroup,
}

impl<R: BufferRaw> SimpleBufferBind<R> {
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

pub struct SimpleBufferBindLayout(wgpu::BindGroupLayout);

impl SimpleBufferBindLayout {
    pub fn new(ctx: &GpuContext, label: Option<&str>) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
