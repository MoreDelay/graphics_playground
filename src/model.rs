//! Definition and handling of 2d objects and their interaction

use crate::instruments::GpuContext;
use crate::instruments::mesh::primitives::{Instances, Triangles, Vertices};
use crate::instruments::mesh::{IndexBuffer, InstanceBuffer, VertexBuffer};

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

/// Index into mesh instancing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshIndex(u32);

impl MeshInstancing {
    /// Create an empty storage for meshes and its instances
    pub const fn new() -> Self {
        Self { meshes: Vec::new() }
    }

    /// Insert a new mesh with its instances to this storage
    pub fn push(&mut self, single: SingleMeshInstancing) -> MeshIndex {
        let index = MeshIndex(self.meshes.len() as u32);
        self.meshes.push(single);
        index
    }

    /// Get a slice of all meshes with their instances
    pub fn slice(&self) -> &[SingleMeshInstancing] {
        &self.meshes
    }

    /// Update the instances for the indexed mesh
    pub fn update_instances(&mut self, ctx: &GpuContext, index: MeshIndex, instances: &Instances) {
        let mesh = &mut self.meshes[index.0 as usize];
        mesh.instances.update(ctx, instances);
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
