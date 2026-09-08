//! Instruments to render simple 2d meshes

pub mod primitives;
pub mod quad;

use std::marker::PhantomData;
use std::num::NonZeroU64;

use iced::wgpu;
use iced::wgpu::util::DeviceExt as _;

use crate::instruments::GpuContext;
use crate::instruments::mesh::primitives::{InstanceRaw, VertexRaw};

/// Interface to upload vertex data in the correct format
pub trait VertexData {
    /// Get a slice to the vertex data
    fn data(&self) -> &[VertexRaw];
}

/// An uploaded vertex buffer ready for use in shaders
pub struct VertexBuffer<T: VertexData> {
    /// The handle to the buffer
    buffer: wgpu::Buffer,
    /// Number of vertices stored
    count: u32,
    /// Marker to associate a specific type of mesh to this vertex buffer
    _marker: PhantomData<T>,
}

impl<T: VertexData> VertexBuffer<T> {
    /// Upload the vertex data by allocating a new buffer
    pub fn upload(ctx: &GpuContext, data: &T) -> Self {
        let data = data.data();
        let count = data.len() as u32;
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::VERTEX,
            });
        Self {
            buffer,
            count,
            _marker: PhantomData,
        }
    }

    /// Get the number of indices stored in this buffer
    #[expect(unused)]
    pub const fn count(&self) -> u32 {
        self.count
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
    /// Number of indices stored
    count: u32,
    /// Marker to associate a specific type of mesh to this index buffer
    _marker: PhantomData<T>,
}

impl<T: IndexData> IndexBuffer<T> {
    /// Upload the indices by allocating a new buffer
    pub fn upload(ctx: &GpuContext, data: &T) -> Self {
        let data = data.data();
        let count = data.len() as u32 * 3;
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::INDEX,
            });
        Self {
            buffer,
            count,
            _marker: PhantomData,
        }
    }

    /// Get the number of indices stored in this buffer
    pub const fn count(&self) -> u32 {
        self.count
    }
}

impl<T: IndexData> std::ops::Deref for IndexBuffer<T> {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

/// Interface to upload vertex data in the correct format
pub trait InstanceData {
    /// Get a slice to the vertex data
    fn data(&self) -> &[InstanceRaw];
}

/// An uploaded vertex buffer ready for use in shaders
pub struct InstanceBuffer<T: InstanceData> {
    /// The handle to the buffer
    buffer: wgpu::Buffer,
    /// Number of instances stored
    count: u32,
    /// Marker to associate a specific type of mesh to this vertex buffer
    _marker: PhantomData<T>,
}

impl<T: InstanceData> InstanceBuffer<T> {
    /// Memory size of a single [`InstanceRaw`]
    const SINGLE_SIZE: NonZeroU64 =
        NonZeroU64::new(std::mem::size_of::<InstanceRaw>() as u64).expect("struct not empty");

    /// Upload the vertex data by allocating a new buffer
    pub fn upload(ctx: &GpuContext, data: &T) -> Self {
        let data = data.data();
        let count = data.len() as u32;
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            });
        Self {
            buffer,
            count,
            _marker: PhantomData,
        }
    }

    /// Get the number of instances stored in this buffer
    pub const fn count(&self) -> u32 {
        self.count
    }

    /// Upload new instance matrices for this mesh
    pub fn update(&self, ctx: &GpuContext, data: &T) {
        let data = data.data();
        let size = data.len() as u64 * Self::SINGLE_SIZE.get();
        let size = NonZeroU64::new(size).expect("empty instance buffer not yet supported");
        ctx.queue
            .write_buffer_with(&self.buffer, 0, size)
            .expect("failed creating temporary buffer for upload")
            .copy_from_slice(bytemuck::cast_slice(data));
    }
}

impl<T: InstanceData> std::ops::Deref for InstanceBuffer<T> {
    type Target = wgpu::Buffer;

    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
