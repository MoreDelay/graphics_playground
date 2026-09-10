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
    pub fn distance(&self, point: na::Point2<f32>) -> f32 {
        let point = point.to_homogeneous();
        self.normal.dot(&point) / self.normal.xy().norm()
    }

    /// Get the normal vector of the dividing line
    pub fn normal(&self) -> na::UnitVector2<f32> {
        na::Unit::new_normalize(self.normal.xy())
    }

    /// Get the point that lies on the line and is closest to the origin
    pub fn closest_to_origin(&self) -> na::Point2<f32> {
        let dist = self.distance(na::Point2::origin());
        let point = -self.normal().into_inner() * dist;
        na::Point2::from(point)
    }

    /// Finds the intersection point of two half spaces
    pub fn intersection(&self, other: &Self) -> Option<na::Point2<f32>> {
        let matrix = na::Matrix3::from_rows(&[
            self.normal.into_inner().transpose(),
            other.normal.into_inner().transpose(),
            na::Vector3::z().transpose(),
        ]);
        let target = na::Vector3::z();
        let solution = na::ColPivQR::new(matrix).solve(&target)?;
        Some(na::Point2::from(solution.xy()))
    }
}

/// A line in 2d space
#[derive(Debug, Clone, Copy)]
pub struct Line(HalfSpace);

impl Line {
    /// Constructor by point and direction
    #[expect(unused)]
    pub fn from_point_and_direction(
        point: na::Point2<f32>,
        direction: na::UnitVector2<f32>,
    ) -> Self {
        Self(HalfSpace::from_point_and_direction(point, direction))
    }

    /// Constructor by two points
    pub fn from_points(p1: na::Point2<f32>, p2: na::Point2<f32>) -> Self {
        Self(HalfSpace::from_points(p1, p2))
    }

    /// Compute the shortest distance between the given point and this line
    #[expect(unused)]
    pub fn distance(&self, point: na::Point2<f32>) -> f32 {
        self.0.distance(point).abs()
    }

    /// Get the point that lies on the line and is closest to the origin
    pub fn closest_to_origin(&self) -> na::Point2<f32> {
        self.0.closest_to_origin()
    }

    /// Finds the intersection point of two lines
    pub fn intersection(&self, other: &Self) -> Option<na::Point2<f32>> {
        self.0.intersection(&other.0)
    }
}

/// A line segment in 2d space
pub struct LineSegment {
    /// The infinite line on which this segment lies
    line: Line,
    /// First end point as distance from the closest point on the line to the origin
    start: f32,
    /// Second end point as distance from the closest point on the line to the origin
    end: f32,
}

impl LineSegment {
    /// Constructor by two points
    #[expect(unused)]
    pub fn from_points(p1: na::Point2<f32>, p2: na::Point2<f32>) -> Self {
        let line = Line::from_points(p1, p2);
        let normal = line.0.normal();
        let dir = Self::direction(normal);
        let mut start = dir.dot(&p1.coords);
        let mut end = dir.dot(&p2.coords);
        if start <= end {
            Self { line, start, end }
        } else {
            let (start, end) = (end, start);
            Self { line, start, end }
        }
    }

    /// Get the start point
    #[expect(unused)]
    pub fn start(&self) -> na::Point2<f32> {
        self.offset_point(self.start)
    }

    /// Get the end point
    #[expect(unused)]
    pub fn end(&self) -> na::Point2<f32> {
        self.offset_point(self.end)
    }

    /// Finds the intersection point of two line segments
    #[expect(unused)]
    pub fn intersecttion(&self, other: &Self) -> Option<na::Point2<f32>> {
        let intersection = self.line.intersection(&other.line)?;
        let offset = self.line_offset(intersection);
        let inside_segment = self.end >= offset && offset >= self.start;
        inside_segment.then_some(intersection)
    }

    /// Compute the direction vector from a line's normal vector
    fn direction(normal: na::UnitVector2<f32>) -> na::UnitVector2<f32> {
        let n = normal.into_inner();
        na::Unit::new_unchecked(na::Vector2::new(-n.y, n.x))
    }

    /// Compute the perpendicular offset to the closest point to the origin
    ///
    /// This is basically the distance of query point to the line defined by the point closest to
    /// the origin and the normal rotated by 90 degrees.
    fn line_offset(&self, point: na::Point2<f32>) -> f32 {
        let normal = self.line.0.normal();
        let dir = Self::direction(normal).into_inner();
        dir.dot(&point.coords)
    }

    /// Offset a point along the line starting from the closest point to the origin
    ///
    /// The offset direction is a 90 degree rotation from the line normal.
    fn offset_point(&self, offset: f32) -> na::Point2<f32> {
        let normal = self.line.0.normal();
        let dir = Self::direction(normal).into_inner();
        let point = self.line.closest_to_origin();
        point + dir * offset
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
