//! Contains the 2d physics simulation widget

use iced::wgpu;
use nalgebra as na;

use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};
use crate::instruments::viewport::Viewport;
use crate::instruments::{GpuContext, TargetContext, Use};

/// The 2d physics simulation widget
pub struct PhysicsWidget {
    /// Rendering instruments of this widget
    instruments: PhysicsInstruments,
    /// The current simulation state
    state: SimulationState,
    /// The size of the visible area
    size: na::Vector2<f32>,
}

impl PhysicsWidget {
    /// Create a new physics widget
    pub fn new() -> Self {
        let size = na::Vector2::new(10., 10.);
        Self {
            instruments: PhysicsInstruments::default(),
            state: SimulationState::init(size),
            size,
        }
    }

    /// Render the current state of objects
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

    /// Handle a message for the physics widget
    pub fn update(&mut self, message: PhysicsMessage) {
        match message {
            PhysicsMessage::Reset => {
                self.state = SimulationState::init(self.size);
            }
            PhysicsMessage::Tick => self.tick(),
        }
    }

    /// Progress the simulation by one tick
    fn tick(&mut self) {
        todo!()
    }
}

/// Messages that can be sent to the [`PhysicsWidget`]
#[derive(Debug, Clone, Copy)]
pub enum PhysicsMessage {
    /// Reset the simulation configuration to its initial state
    Reset,
    /// Progress the simulation by one tick
    Tick,
}

/// The state of the physics simulation
struct SimulationState {
    /// The objects that do not move
    fixed: Vec<FixedBody>,
    /// The objects that may move due to force interactions
    moving: Vec<MovingBody>,
}

impl SimulationState {
    /// Create the initial state of the hard-coded scenario
    fn init(size: na::Vector2<f32>) -> Self {
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

        let shape = MovingShape::Rectangle(Rectangle {
            size: na::Vector2::new(1., 1.),
            rotation: 0.,
            center: na::Point2::new(2., 7.),
        });
        let square = MovingBody {
            shape,
            velocity: na::Vector2::zeros(),
            angular_velocity: 0.,
        };
        let moving = vec![square];

        Self { fixed, moving }
    }
}

/// All different types of unmoving shapes
#[derive(Debug, Clone)]
enum FixedBody {
    /// A fixed half space
    HalfSpace(HalfSpace),
    /// A fixed rectangle
    Square(Rectangle),
}

/// State of a moving object
#[derive(Debug, Clone)]
struct MovingBody {
    /// The shape of this object
    shape: MovingShape,
    /// The current velocity
    velocity: na::Vector2<f32>,
    /// The current counter-clockwise angular velocity
    angular_velocity: f32,
}

/// All different types of movable shapes
#[derive(Debug, Clone)]
enum MovingShape {
    /// A movable rectangle
    Rectangle(Rectangle),
}

/// A half space, splitting the whole space in inner and outer half
#[derive(Debug, Clone)]
struct HalfSpace {
    /// The normal direction of the split line
    normal: na::Vector2<f32>,
    /// The signed distance from the origin to the split line
    distance: f32,
}

/// A rectangle
#[derive(Debug, Clone)]
pub struct Rectangle {
    /// The size of the rectangle in x and y dimension
    size: na::Vector2<f32>,
    /// The counter-clockwise rotation
    rotation: f32,
    /// The location of the centroid
    center: na::Point2<f32>,
}

impl Rectangle {
    /// Access the size of this rectangle
    pub const fn size(&self) -> na::Vector2<f32> {
        self.size
    }
}

/// The rendering instruments for the physics simulation
#[derive(Default)]
struct PhysicsInstruments {
    /// The last rendering output given out
    final_output: Use<PassThruTexture>,
}
