use std::marker::PhantomData;
use std::num::NonZeroU64;

use iced::wgpu;
use iced::wgpu::util::DeviceExt as _;

use crate::gpu::GpuContext;

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
