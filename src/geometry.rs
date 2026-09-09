//! Contains structs that represent geometric forms

use nalgebra as na;

use crate::instruments::mesh::primitives::{InstanceRaw, Triangles, VertexRaw, Vertices};
use crate::model::MeshCpu;

/// A line dividing 2d space, represented in homogeneous coordinates
#[derive(Debug, Clone, Copy)]
pub struct HalfSpace {
    /// The plane normal in 3d that describes a line by its intersection with the plane Z=1
    normal: na::UnitVector3<f32>,
}

impl HalfSpace {
    /// Constructor by point and direction
    ///
    /// From the perspective of the direction, "inside" is to the left. In other words, a rotation
    /// of (0, 180) degrees (excluding ends) of the direction vector makes it a vector that points
    /// towards the inside.
    pub fn from_point_and_direction(
        point: na::Point2<f32>,
        direction: na::UnitVector2<f32>,
    ) -> Self {
        let point = point.to_homogeneous();
        let direction = direction.to_homogeneous();
        let normal = na::Unit::new_normalize(point.cross(&direction));
        Self { normal }
    }

    /// Constructor by two points
    ///
    /// This is the same as [`Self::from_point_and_direction`], using the direction `p2 - p1`.
    pub fn from_points(p1: na::Point2<f32>, p2: na::Point2<f32>) -> Self {
        assert_ne!(p1, p2, "can not create line from two identical points");
        let dir = na::Unit::new_normalize(p2 - p1);
        Self::from_point_and_direction(p1, dir)
    }

    /// Compute the signed shortest distance between the given point and the dividing line
    ///
    /// A negative distance indicates the point lies on the inside of this half space.
    #[expect(unused)]
    pub fn distance(&self, point: na::Point2<f32>) -> f32 {
        let point = point.to_homogeneous();
        self.normal.dot(&point) / self.normal.xy().norm()
    }
}

/// A rectangle
#[derive(Debug, Clone, Copy)]
pub struct Rectangle {
    /// The size of the rectangle in x and y dimension
    size: na::Vector2<f32>,
    /// The counter-clockwise rotation
    rotation: f32,
    /// The location of the centroid
    center: na::Point2<f32>,
}

impl Rectangle {
    /// Create a new rectangle with the provided size and pose
    pub const fn new(size: na::Vector2<f32>, rotation: f32, center: na::Point2<f32>) -> Self {
        Self {
            size,
            rotation,
            center,
        }
    }

    /// Transform the pose of this rectangle
    pub fn transform(&mut self, translation: na::Vector2<f32>, angle: f32) {
        self.center += translation;
        self.rotation += angle;
    }

    /// Create the base mesh for a rectangle
    ///
    /// This mesh has its vertices at x, y in {-1, 1}. These should be transformed by the instance
    /// matrix to align with the expected shape.
    pub fn mesh() -> MeshCpu {
        let red = [1., 0., 0.];
        let green = [0., 1., 0.];
        let blue = [0., 0., 1.];
        let white = [1., 1., 1.];

        let nw = VertexRaw::new([-1., 1.], [0., 0.], red);
        let ne = VertexRaw::new([1., 1.], [1., 0.], green);
        let sw = VertexRaw::new([-1., -1.], [0., 1.], blue);
        let se = VertexRaw::new([1., -1.], [1., 1.], white);

        let vertices = vec![nw, ne, sw, se];
        let vertices = Vertices::new(vertices);

        let tri0 = na::Vector3::new(0, 2, 1);
        let tri1 = na::Vector3::new(1, 2, 3);
        let indices = vec![tri0, tri1];
        let indices = Triangles::new(indices);

        MeshCpu::new(vertices, indices)
    }

    /// Create an instance transform for this rectangle
    pub fn instance(&self) -> InstanceRaw {
        let stretch = na::Matrix2::from_diagonal(&(self.size / 2.));
        let transform = na::Rotation2::new(self.rotation) * stretch;

        let model0 = transform.column(0).to_homogeneous();
        let model1 = transform.column(1).to_homogeneous();
        let model2 = na::Point2::from(self.center).to_homogeneous();

        let model0 = model0.into();
        let model1 = model1.into();
        let model2 = model2.into();
        InstanceRaw::new(model0, model1, model2)
    }
}
