//! Definition and handling of 2d objects and their interaction

use nalgebra as na;

use crate::instruments::GpuContext;
use crate::instruments::mesh::{IndexBuffer, IndexData, VertexBuffer, VertexData, VertexRaw};
use crate::physics::Rectangle;

/// An array-of-structs of vertex data
pub struct Vertices(Vec<VertexRaw>);

impl VertexData for Vertices {
    fn data(&self) -> &[VertexRaw] {
        &self.0
    }
}

/// An array of triangles indexing into some [`Vertices`] array
pub struct Triangles(Vec<na::Vector3<u32>>);

impl IndexData for Triangles {
    fn data(&self) -> &[[u32; 3]] {
        bytemuck::cast_slice(&self.0)
    }
}

/// A simple 2d mesh representation that can be uploaded for rendering
pub struct MeshCpu {
    /// The vertices of this mesh
    vertices: Vertices,
    /// The triangle of this mesh, indexing into the vertices
    indices: Triangles,
}

impl MeshCpu {
    /// Create a new mesh for a rectangle
    pub fn new_rect(rect: &Rectangle) -> Self {
        let size = rect.size();
        let half = na::Vector2::new(size.x / 2., size.y / 2.);
        let color = [1., 1., 1.];
        let nw = VertexRaw::new([-half.x, half.y], color);
        let ne = VertexRaw::new([half.x, half.y], color);
        let sw = VertexRaw::new([-half.x, -half.y], color);
        let se = VertexRaw::new([half.x, -half.y], color);
        let vertices = vec![nw, ne, sw, se];
        let vertices = Vertices(vertices);

        let tri0 = na::Vector3::new(0, 1, 2);
        let tri1 = na::Vector3::new(2, 1, 3);
        let indices = vec![tri0, tri1];
        let indices = Triangles(indices);

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
}
