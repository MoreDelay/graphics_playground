//! Defines structs for primitve data types for defining meshes, such as vertices or indices.

use iced::wgpu;
use nalgebra as na;

use crate::instruments::mesh::{IndexData, InstanceData, VertexData};

/// The format expected by shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexRaw {
    /// The position coordinates of this vertex
    pub pos: [f32; 2],
    /// UV coordinates at this vertex
    pub uv: [f32; 2],
    /// The color of this vertex
    pub color: [f32; 3],
    /// Padding to satisfy wgpu alignment constraints
    pub _pad: u32,
}

impl VertexRaw {
    /// Create a new raw vertex
    ///
    /// Just a helper to eliminate noise about the padding
    pub const fn new(pos: [f32; 2], uv: [f32; 2], color: [f32; 3]) -> Self {
        Self {
            pos,
            uv,
            color,
            _pad: 0,
        }
    }

    /// Create the vertex descriptor for these instances
    pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// The raw data layout for an instance, passed as vertex attributes
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    /// column 0 of homogeneous transformation matrix in world space
    pub model0: [f32; 3],
    /// Padding 0
    pub _pad0: u32,
    /// column 1 of homogeneous transformation matrix in world space
    pub model1: [f32; 3],
    /// Padding 1
    pub _pad1: u32,
    /// column 2 of homogeneous transformation matrix in world space
    pub model2: [f32; 3],
    /// Padding 2
    pub _pad2: u32,
}

impl InstanceRaw {
    /// Create a new raw instance
    ///
    /// Just a helper to eliminate noise about the padding. Vectors are column vectors of the
    /// homogeneous transformation matrix.
    pub const fn new(model0: [f32; 3], model1: [f32; 3], model2: [f32; 3]) -> Self {
        Self {
            model0,
            _pad0: 0,
            model1,
            _pad1: 0,
            model2,
            _pad2: 0,
        }
    }

    /// Create the vertex descriptor for these instances
    pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

impl InstanceData for InstanceRaw {
    fn data(&self) -> &[InstanceRaw] {
        std::slice::from_ref(self)
    }
}

/// An array-of-structs of vertex data
#[derive(Debug)]
pub struct Vertices(Vec<VertexRaw>);

impl Vertices {
    /// Wrap raw vertices into an vertex array
    pub fn new(data: Vec<VertexRaw>) -> Self {
        assert!(!data.is_empty(), "empty vertex array not allowed");
        Self(data)
    }
}

impl VertexData for Vertices {
    fn data(&self) -> &[VertexRaw] {
        &self.0
    }
}

/// An array of triangles indexing into some [`Vertices`] array
#[derive(Debug)]
pub struct Triangles(Vec<na::Vector3<u32>>);

impl Triangles {
    /// Wrap triangle indices into a triangles index array
    pub fn new(data: Vec<na::Vector3<u32>>) -> Self {
        assert!(!data.is_empty(), "empty index array not allowed");
        Self(data)
    }
}

impl IndexData for Triangles {
    fn data(&self) -> &[[u32; 3]] {
        bytemuck::cast_slice(&self.0)
    }
}

/// An array-of-structs of vertex data
pub struct Instances(Vec<InstanceRaw>);

impl Instances {
    /// Wrap raw instance transforms into an instance array
    pub const fn new(data: Vec<InstanceRaw>) -> Self {
        assert!(!data.is_empty(), "empty instance array not allowed");
        Self(data)
    }
}

impl InstanceData for Instances {
    fn data(&self) -> &[InstanceRaw] {
        &self.0
    }
}
