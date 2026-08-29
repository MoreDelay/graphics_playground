use nalgebra as na;

use crate::instruments::mesh::VertexData;

pub struct Positions(Vec<na::Vector2<f32>>);

impl VertexData for Positions {
    fn data(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

pub struct Triangles(Vec<na::Vector3<u32>>);

impl VertexData for Triangles {
    fn data(&self) -> &[u8] {
        bytemuck::cast_slice(&self.0)
    }
}

pub struct Mesh {
    positions: Positions,
    triangles: Triangles,
}
