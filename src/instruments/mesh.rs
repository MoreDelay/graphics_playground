//! Instruments to render simple 2d meshes

use std::marker::PhantomData;

use iced::wgpu;
use iced::wgpu::util::DeviceExt as _;

use crate::instruments::GpuContext;

/// Interface to upload vertex data in the correct format
pub trait VertexData {
    /// Get a slice to the vertex data
    fn data(&self) -> &[VertexRaw];
}

/// An uploaded vertex buffer ready for use in shaders
pub struct VertexBuffer<T: VertexData> {
    /// The handle to the buffer
    buffer: wgpu::Buffer,
    /// Marker to associate a specific type of mesh to this vertex buffer
    _marker: PhantomData<T>,
}

impl<T: VertexData> VertexBuffer<T> {
    /// Upload the vertex data by allocating a new buffer
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

/// Interface to upload triangle indices
pub trait IndexData {
    /// Get a slice to the triangle indices
    fn data(&self) -> &[[u32; 3]];
}

/// An uploaded index buffer ready for use in shaders
pub struct IndexBuffer<T: IndexData> {
    /// The handle to the buffer
    buffer: wgpu::Buffer,
    /// Marker to associate a specific type of mesh to this index buffer
    _marker: PhantomData<T>,
}

impl<T: IndexData> IndexBuffer<T> {
    /// Upload the indices by allocating a new buffer
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

/// The format expected by shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexRaw {
    /// The position coordinates of this vertex
    pub pos: [f32; 2],
    /// Padding to satisfy wgpu alignment constraints
    pub _pad1: [u32; 2],
    /// The color of this vertex
    pub color: [f32; 3],
    /// Padding to satisfy wgpu alignment constraints
    pub _pad2: u32,
}

impl VertexRaw {
    /// Create a new raw vertex
    ///
    /// Just a helper to eliminate noise about the padding
    pub const fn new(pos: [f32; 2], color: [f32; 3]) -> Self {
        Self {
            pos,
            _pad1: [0, 0],
            color,
            _pad2: 0,
        }
    }
}
