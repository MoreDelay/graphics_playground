//! Definition and handling of 2d objects and their interaction

use nalgebra as na;

use crate::instruments::GpuContext;
use crate::instruments::bind::physics::{InstanceRaw, VertexRaw};
use crate::instruments::mesh::{
    IndexBuffer,
    IndexData,
    InstanceBuffer,
    InstanceData,
    VertexBuffer,
    VertexData,
};

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

/// A simple 2d mesh representation that can be uploaded for rendering
#[derive(Debug)]
pub struct MeshCpu {
    /// The vertices of this mesh
    vertices: Vertices,
    /// The triangle of this mesh, indexing into the vertices
    indices: Triangles,
}

impl MeshCpu {
    /// Create a new mesh
    pub const fn new(vertices: Vertices, indices: Triangles) -> Self {
        Self { vertices, indices }
    }

    /// Upload this mesh to the gpu
    pub fn upload(&self, ctx: &GpuContext) -> MeshGpu {
        MeshGpu::upload(ctx, self)
    }
}

/// A simple 2d mesh representation uploaded and ready for rendering
pub struct MeshGpu {
    /// The vertex buffer of this mesh
    vertices: VertexBuffer<Vertices>,
    /// The index buffer of this mesh
    indices: IndexBuffer<Triangles>,
}

impl MeshGpu {
    /// Upload a mesh to the gpu
    pub fn upload(ctx: &GpuContext, mesh: &MeshCpu) -> Self {
        let vertices = VertexBuffer::upload(ctx, &mesh.vertices);
        let indices = IndexBuffer::upload(ctx, &mesh.indices);
        Self { vertices, indices }
    }

    /// Get the vertex buffer for this mesh
    pub const fn vertices(&self) -> &VertexBuffer<Vertices> {
        &self.vertices
    }

    /// Get the index buffer for this mesh
    pub const fn indices(&self) -> &IndexBuffer<Triangles> {
        &self.indices
    }
}

/// Holds all meshes together with their instances
pub struct MeshInstancing {
    /// The instanced meshes
    meshes: Vec<SingleMeshInstancing>,
}

impl MeshInstancing {
    /// Create an empty storage for meshes and its instances
    pub const fn new() -> Self {
        Self { meshes: Vec::new() }
    }

    /// Insert a new mesh with its instances to this storage
    pub fn push(&mut self, single: SingleMeshInstancing) {
        self.meshes.push(single);
    }

    /// Get a slice of all meshes with their instances
    pub fn slice(&self) -> &[SingleMeshInstancing] {
        &self.meshes
    }
}

/// A single base mesh together with all its instances
pub struct SingleMeshInstancing {
    /// The base geometry for these instances
    base: MeshGpu,
    /// The instance transforms
    instances: InstanceBuffer<Instances>,
}

impl SingleMeshInstancing {
    /// Define a new renderable mesh with its instances
    pub const fn new(base: MeshGpu, instances: InstanceBuffer<Instances>) -> Self {
        Self { base, instances }
    }

    /// Get a reference to the base mesh
    pub const fn base(&self) -> &MeshGpu {
        &self.base
    }

    /// Get a reference to the instances of this mesh
    pub const fn instances(&self) -> &InstanceBuffer<Instances> {
        &self.instances
    }
}
