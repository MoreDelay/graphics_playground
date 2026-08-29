use std::marker::PhantomData;

use iced::wgpu;
use iced::wgpu::util::DeviceExt as _;

use crate::instruments::GpuContext;

pub trait VertexData {
    fn data(&self) -> &[u8];
}

struct VertexBuffer<T: VertexData> {
    buffer: wgpu::Buffer,
    _marker: PhantomData<T>,
}

impl<T: VertexData> VertexBuffer<T> {
    pub fn upload(ctx: &GpuContext, data: &T) -> Self {
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data.data()),
                usage: wgpu::BufferUsages::VERTEX,
            });
        Self {
            buffer,
            _marker: PhantomData,
        }
    }
}

impl<T: VertexData> std::ops::Deref for VertexBuffer<T> {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

pub trait IndexData {
    fn data(&self) -> &[u8];
}

struct IndexBuffer<T: IndexData> {
    buffer: wgpu::Buffer,
    _marker: PhantomData<T>,
}

impl<T: IndexData> IndexBuffer<T> {
    pub fn upload(ctx: &GpuContext, data: &T) -> Self {
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data.data()),
                usage: wgpu::BufferUsages::INDEX,
            });
        Self {
            buffer,
            _marker: PhantomData,
        }
    }
}

impl<T: IndexData> std::ops::Deref for IndexBuffer<T> {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
