//! Defines the simple quad mesh used for drawing 2d scenes

use nalgebra as na;

use crate::geometry::Rectangle;
use crate::instruments::GpuContext;
use crate::instruments::mesh::primitives::InstanceRaw;
use crate::model::MeshGpu;

/// A quad mesh
///
/// Used for 2d objects that can be represented as a simple quad, such as the displayed image in the
/// image viewer. This quad comes together with an instance transformation that adapts the vertices
/// as needed.
pub struct QuadMesh {
    /// The quad mesh
    mesh: MeshGpu,
}

impl QuadMesh {
    /// Create a new quad mesh, ready for rendering
    pub fn new(ctx: &GpuContext) -> Self {
        let mesh = Rectangle::mesh().upload(ctx);
        Self { mesh }
    }

    /// Create a box instance
    ///
    /// This instance transforms the box such that one corner lies at the origin, and the opposite
    /// corner lies where `size` points to.
    pub fn box_instance(size: na::Vector2<f32>) -> InstanceRaw {
        let rect = Rectangle::new(size, 0., na::Point2::origin());
        rect.instance()
    }
}

impl std::ops::Deref for QuadMesh {
    type Target = MeshGpu;

    fn deref(&self) -> &Self::Target {
        &self.mesh
    }
}
