use iced::wgpu;
use nalgebra as na;

use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};
use crate::instruments::viewport::Viewport;
use crate::instruments::{GpuContext, TargetContext};

pub struct PhysicsWidget {
    fixed: Vec<FixedBody>,
    moving: Vec<MovingBody>,

    size: na::Vector2<f32>,
}

impl PhysicsWidget {
    pub fn new() -> Self {
        let size = na::Vector2::new(10., 10.);

        let edge_left = FixedBody::HalfSpace(HalfSpace {
            normal: na::Vector2::new(-1., 0.),
            distance: 0.,
        });
        let edge_right = FixedBody::HalfSpace(HalfSpace {
            normal: na::Vector2::new(1., 0.),
            distance: size.x,
        });
        let edge_top = FixedBody::HalfSpace(HalfSpace {
            normal: na::Vector2::new(0., 1.),
            distance: size.y,
        });
        let edge_bottom = FixedBody::HalfSpace(HalfSpace {
            normal: na::Vector2::new(0., -1.),
            distance: 0.,
        });
        let fixed = vec![edge_left, edge_right, edge_top, edge_bottom];

        let square = MovingBody::Square(Rectangle {
            size: na::Vector2::new(1., 1.),
            rotation: 0.,
            center: na::Point2::new(2., 7.),
        });
        let moving = vec![square];

        Self {
            fixed,
            moving,
            size,
        }
    }

    pub fn render(
        &mut self,
        ctx: &GpuContext,
        target: &TargetContext,
        passthru: &PassThruPipeline,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
    ) -> Option<&PassThruTexture> {
        None
    }

    pub fn update(&mut self, message: PhysicsMessage) {
        todo!()
    }
}

#[derive(Debug, Clone, Copy)]
pub enum PhysicsMessage {
    Reset,
}

enum FixedBody {
    HalfSpace(HalfSpace),
    Square(Rectangle),
}

enum MovingBody {
    Square(Rectangle),
}

struct HalfSpace {
    normal: na::Vector2<f32>,
    distance: f32,
}

struct Rectangle {
    size: na::Vector2<f32>,
    rotation: f32,
    center: na::Point2<f32>,
}
