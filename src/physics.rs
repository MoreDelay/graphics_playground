//! Contains the 2d physics simulation widget

use iced::wgpu;
use nalgebra as na;

use crate::instruments::bind::physics::{CameraInfoRaw, InstanceRaw, VertexRaw};
use crate::instruments::buffer::{SimpleBuffer, SimpleBufferBind, SimpleBufferBindLayout};
use crate::instruments::mesh::InstanceBuffer;
use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};
use crate::instruments::pipeline::physics::{PhysicsObjectPipeline, PhysicsObjectPipelineLayout};
use crate::instruments::viewport::Viewport;
use crate::instruments::{GpuContext, TargetContext, Use};
use crate::model::{Instances, MeshCpu, MeshInstancing, SingleMeshInstancing, Triangles, Vertices};

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
    pub fn new(ctx: &GpuContext, target: &TargetContext) -> Self {
        let size = na::Vector2::new(10., 10.);
        let state = SimulationState::init(size);
        let rect_instances: Vec<_> = state
            .moving
            .iter()
            .map(|obj| match &obj.shape {
                MovingShape::Rectangle(rect) => rect.instance(),
            })
            .collect();

        let rect_instances = Instances::new(rect_instances);
        let rect_instances = InstanceBuffer::upload(ctx, &rect_instances);
        let mut instancing = MeshInstancing::new();
        let rect = Rectangle::mesh().upload(ctx);
        let rect = SingleMeshInstancing::new(rect, rect_instances);
        instancing.push(rect);
        let instruments = PhysicsInstruments::new(ctx, target, instancing, size);

        Self {
            instruments,
            state,
            size,
        }
    }

    /// Render the current state of objects
    pub fn render(
        &mut self,
        ctx: &GpuContext,
        passthru: &PassThruPipeline,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
    ) -> Option<&PassThruTexture> {
        if self.instruments.final_output().is_some() {
            return self.instruments.final_output();
        }

        let output = self.instruments.create_output(ctx, passthru, viewport)?;

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Physics Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output.view(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        self.instruments.pipeline.draw(
            &mut pass,
            &self.instruments.camera,
            &self.instruments.moving,
        );

        Some(self.instruments.set_final_output(output))
    }

    /// Handle a message for the physics widget
    pub fn update(&mut self, message: PhysicsMessage) {
        match message {
            PhysicsMessage::Reset => {
                self.instruments.final_output.degrade();
                self.state = SimulationState::init(self.size);
            }
            PhysicsMessage::Tick => self.tick(),
        }
    }

    /// Progress the simulation by one tick
    fn tick(&mut self) {
        self.instruments.final_output.degrade();
        // TODO: doing nothing yet
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
    #[expect(unused)]
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
            size: na::Vector2::new(0.5, 0.3),
            rotation: 0.,
            center: na::Point2::new(1.0, 1.0),
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
    #[expect(unused)]
    HalfSpace(HalfSpace),
    /// A fixed rectangle
    #[expect(unused)]
    Rectangle(Rectangle),
}

/// State of a moving object
#[derive(Debug, Clone)]
struct MovingBody {
    /// The shape of this object
    shape: MovingShape,
    /// The current velocity
    #[expect(unused)]
    velocity: na::Vector2<f32>,
    /// The current counter-clockwise angular velocity
    #[expect(unused)]
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
    #[expect(unused)]
    normal: na::Vector2<f32>,
    /// The signed distance from the origin to the split line
    #[expect(unused)]
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
    /// Create the base mesh for a rectangle
    ///
    /// This mesh has its vertices at x, y in {-1, 1}. These should be transformed by the instance
    /// matrix to align with the expected shape.
    pub fn mesh() -> MeshCpu {
        let red = [1., 0., 0.];
        let green = [0., 1., 0.];
        let blue = [0., 0., 1.];
        let white = [1., 1., 1.];

        let nw = VertexRaw::new([-1., 1.], red);
        let ne = VertexRaw::new([1., 1.], green);
        let sw = VertexRaw::new([-1., -1.], blue);
        let se = VertexRaw::new([1., -1.], white);

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
        let stretch = na::Matrix2::from_diagonal(&self.size);
        let transform = na::Rotation2::new(self.rotation) * stretch;
        let model0 = transform.column(0).to_homogeneous();
        let model1 = transform.column(1).to_homogeneous();
        let model2 = na::Vector3::new(self.center.x, self.center.y, 1.);
        let model0 = bytemuck::cast(model0);
        let model1 = bytemuck::cast(model1);
        let model2 = bytemuck::cast(model2);
        InstanceRaw::new(model0, model1, model2)
    }
}

/// The rendering instruments for the physics simulation
struct PhysicsInstruments {
    /// The last rendering output given out
    final_output: Use<PassThruTexture>,

    /// The render pipeline
    pipeline: PhysicsObjectPipeline,
    /// Objects influenced by physics
    moving: MeshInstancing,
    /// Buffer holding the camera location
    camera: SimpleBufferBind<CameraInfoRaw>,
}

impl PhysicsInstruments {
    /// Create a new set of rendering instruments for the physics widget
    fn new(
        ctx: &GpuContext,
        target: &TargetContext,
        moving: MeshInstancing,
        size: na::Vector2<f32>,
    ) -> Self {
        let buffer_layout = SimpleBufferBindLayout::new(ctx, Some("Physics Buffer Layout"));
        let layout = PhysicsObjectPipelineLayout::new(ctx, &buffer_layout);
        let pipeline = PhysicsObjectPipeline::new(ctx, &layout, target.config.format);

        let camera = CameraInfoRaw {
            start: [0., 0.],
            size: size.into(),
        };
        let camera = SimpleBuffer::new(ctx, camera, Some("Physics Camera Buffer"));
        let camera = SimpleBufferBind::new(
            ctx,
            camera,
            &buffer_layout,
            Some("Physics Camera Buffer Bind"),
        );

        Self {
            final_output: Use::default(),
            pipeline,
            moving,
            camera,
        }
    }

    /// Get the current final output
    pub const fn final_output(&self) -> Option<&PassThruTexture> {
        let Use::Active(output) = &self.final_output else {
            return None;
        };
        Some(output)
    }

    /// Get the current final output
    pub fn set_final_output(&mut self, output: PassThruTexture) -> &PassThruTexture {
        self.final_output = Use::Active(output);
        self.final_output.active()
    }

    /// Extract or create the render target for the widget
    pub fn create_output(
        &mut self,
        ctx: &GpuContext,
        passthru: &PassThruPipeline,
        viewport: &Viewport,
    ) -> Option<PassThruTexture> {
        let Some(extent) = viewport.extent() else {
            self.final_output = self.final_output.take().make_unused();
            return None;
        };

        match self.final_output.take() {
            Use::Invalid => {
                self.final_output = Use::Invalid;
                None
            }
            Use::Active(output) | Use::Recycle(output) | Use::Unused(output)
                if output.texture().size() == extent =>
            {
                Some(output)
            }
            Use::Active(_) | Use::Recycle(_) | Use::Unused(_) | Use::Missing => {
                Some(passthru.create_texture(ctx, extent))
            }
        }
    }
}
