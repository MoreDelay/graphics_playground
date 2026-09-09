//! Contains the 2d physics simulation widget

use std::time::{Duration, Instant};

use iced::wgpu;
use iced_winit::winit::dpi::PhysicalSize;
use nalgebra as na;

use crate::controls::RenderContext;
use crate::geometry::{HalfSpace, Rectangle};
use crate::instruments::bind::image::ViewportRaw;
use crate::instruments::buffer::{
    SimpleBuffer,
    SimpleBufferBind,
    SimpleBufferBindLayout,
    VisibleVertex,
};
use crate::instruments::mesh::InstanceBuffer;
use crate::instruments::mesh::primitives::Instances;
use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};
use crate::instruments::pipeline::physics::{PhysicsObjectPipeline, PhysicsObjectPipelineLayout};
use crate::instruments::{GpuContext, TargetContext, Use};
use crate::model::{MeshIndex, MeshInstancing, SingleMeshInstancing};
use crate::viewport::{ScrollableViewportState, ViewportGui, ViewportMessage};

/// Messages that can be sent to the [`PhysicsWidget`]
#[derive(Debug, Clone, Copy)]
pub enum PhysicsMessage {
    /// Reset the simulation configuration to its initial state
    Reset,
    /// Update the visible area of the viewport
    Viewport(ViewportMessage),
    /// Yoink up the box
    YoinkUp,
    /// Yoink down the box
    YoinkDown,
}

/// The 2d physics simulation widget
pub struct PhysicsWidget {
    /// Rendering instruments of this widget
    instruments: PhysicsInstruments,
    /// The current simulation state
    state: SimulationState,
    /// The state of the viewport that is currently shown
    viewport: ScrollableViewportState,
    /// Last time the simulation did progress
    last_tick: Option<Instant>,
}

impl PhysicsWidget {
    /// Create a new physics widget
    pub fn new(ctx: &GpuContext, target: &TargetContext) -> Self {
        let size = na::Vector2::new(2., 2.);
        let view = PhysicalSize::new(1, 1);
        let viewport = ScrollableViewportState::new(size, view, 80.);

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
        let index_temp = instancing.push(rect);
        let instruments = PhysicsInstruments::new(ctx, target, instancing, index_temp);

        Self {
            instruments,
            state,
            viewport,
            last_tick: None,
        }
    }

    /// Render the current state of objects
    pub fn render(&mut self, context: &mut RenderContext) -> Option<&PassThruTexture> {
        self.resize_viewport(context.viewport.size());

        if self.instruments.final_output().is_some() {
            return self.instruments.final_output();
        }

        let RenderContext {
            ctx,
            passthru,
            encoder,
            viewport,
            ..
        } = context;

        self.instruments.create_camera(ctx, &self.viewport);
        self.instruments.update_instances(ctx, &self.state);

        let output = self.instruments.create_output(ctx, passthru, viewport)?;
        let camera = self.instruments.camera.active();

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

        self.instruments
            .pipeline
            .draw(&mut pass, camera, &self.instruments.moving);

        Some(self.instruments.set_final_output(output))
    }

    /// Handle a message for the physics widget
    pub fn update(&mut self, message: PhysicsMessage) {
        self.instruments.final_output.degrade();
        match message {
            PhysicsMessage::Reset => {
                self.instruments.camera.degrade();
                self.viewport.reset();
                self.state = SimulationState::init(self.viewport.area());
            }
            PhysicsMessage::Viewport(message) => {
                self.instruments.camera.degrade();
                self.viewport.update(message);
            }
            PhysicsMessage::YoinkUp => {
                for obj in &mut self.state.moving {
                    obj.yoink_up();
                }
            }
            PhysicsMessage::YoinkDown => {
                for obj in &mut self.state.moving {
                    obj.yoink_down();
                }
            }
        }
    }

    /// Update the viewport size
    fn resize_viewport(&mut self, size: PhysicalSize<u32>) {
        let changed = self.viewport.resize_view(size);
        if !changed {
            return;
        }

        self.instruments.resized();
    }

    /// Progress the simulation by one tick
    ///
    /// Returns the next time a tick can be triggered
    pub fn tick(&mut self) -> Instant {
        const TIME_DELTA: Duration = Duration::from_micros(1_000_000 / 60);

        let now = Instant::now();
        let Some(last_tick) = self.last_tick else {
            self.last_tick = Some(now);
            return now + TIME_DELTA;
        };
        let next = last_tick + TIME_DELTA;
        if now < next {
            return next;
        }
        self.last_tick = Some(now);
        let next = now + TIME_DELTA;

        self.instruments.final_output.degrade();
        for obj in &mut self.state.moving {
            obj.tick();
        }

        next
    }
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
        let bot_lef = na::Point2::origin();
        let top_lef = na::Point2::new(0., size.y);
        let bot_rig = na::Point2::new(size.x, 0.);
        let top_rig = na::Point2::new(size.x, size.y);
        let edge_left = FixedBody::HalfSpace(HalfSpace::from_points(bot_lef, top_lef));
        let edge_top = FixedBody::HalfSpace(HalfSpace::from_points(top_lef, top_rig));
        let edge_right = FixedBody::HalfSpace(HalfSpace::from_points(top_rig, bot_rig));
        let edge_bottom = FixedBody::HalfSpace(HalfSpace::from_points(bot_rig, bot_lef));
        let fixed = vec![edge_left, edge_right, edge_top, edge_bottom];

        let size = na::Vector2::new(0.5, 0.3);
        let rotation = 0.;
        let center = na::Point2::new(1.0, 1.0);
        let square = MovingBody {
            shape: MovingShape::Rectangle(Rectangle::new(size, rotation, center)),
            velocity: na::Vector2::new(0., 5.),
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
    velocity: na::Vector2<f32>,
    /// The current counter-clockwise angular velocity
    angular_velocity: f32,
}

impl MovingBody {
    /// Hard-coded step size of the physics simulation
    const STEP: f32 = 0.01;
    /// The default gravity forced applied on each step
    const GRAVITY: na::Vector2<f32> = na::Vector2::new(0., -9.8);

    /// Yoink up the object
    fn yoink_up(&mut self) {
        self.velocity += na::Vector2::new(0., 4.);
    }

    /// Yoink down the object
    fn yoink_down(&mut self) {
        self.velocity += na::Vector2::new(0., -4.);
    }

    /// Move this object along by a single tick
    fn tick(&mut self) {
        match &mut self.shape {
            MovingShape::Rectangle(rect) => {
                let translation = Self::STEP * self.velocity;
                let angle = Self::STEP * self.angular_velocity;
                rect.transform(translation, angle);
            }
        }

        self.velocity += Self::STEP * Self::GRAVITY;
    }
}

/// All different types of movable shapes
#[derive(Debug, Clone, Copy)]
enum MovingShape {
    /// A movable rectangle
    Rectangle(Rectangle),
}

/// The rendering instruments for the physics simulation
struct PhysicsInstruments {
    /// The last rendering output given out
    final_output: Use<PassThruTexture>,

    /// The render pipeline
    pipeline: PhysicsObjectPipeline,
    /// Objects influenced by physics
    moving: MeshInstancing,
    /// Index returned by [`MeshInstancing`]
    index_temp: MeshIndex,
    /// Buffer holding the camera location
    camera: Use<SimpleBufferBind<ViewportRaw, VisibleVertex>>,
}

impl PhysicsInstruments {
    /// Create a new set of rendering instruments for the physics widget
    fn new(
        ctx: &GpuContext,
        target: &TargetContext,
        moving: MeshInstancing,
        index_temp: MeshIndex,
    ) -> Self {
        let buffer_layout = SimpleBufferBindLayout::new(ctx, Some("Physics Buffer Layout"));
        let layout = PhysicsObjectPipelineLayout::new(ctx, &buffer_layout);
        let pipeline = PhysicsObjectPipeline::new(ctx, &layout, target.config.format);

        let camera = Use::default();

        Self {
            final_output: Use::default(),
            pipeline,
            moving,
            index_temp,
            camera,
        }
    }

    /// Signal that the viewport size has changed
    fn resized(&mut self) {
        self.final_output.degrade();
        self.camera.degrade();
    }

    /// Get the current final output
    const fn final_output(&self) -> Option<&PassThruTexture> {
        let Use::Active(output) = &self.final_output else {
            return None;
        };
        Some(output)
    }

    /// Get the current final output
    fn set_final_output(&mut self, output: PassThruTexture) -> &PassThruTexture {
        self.final_output = Use::Active(output);
        self.final_output.active()
    }

    /// Extract or create the render target for the widget
    fn create_output(
        &mut self,
        ctx: &GpuContext,
        passthru: &PassThruPipeline,
        viewport: &ViewportGui,
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

    /// Create or update the camera transform
    fn create_camera(&mut self, ctx: &GpuContext, viewport: &ScrollableViewportState) {
        let camera = match self.camera.take() {
            Use::Missing | Use::Invalid => {
                let layout = SimpleBufferBindLayout::new(ctx, Some("Physics Buffer Layout"));
                let camera = viewport.as_raw();
                let camera = SimpleBuffer::new(ctx, camera, Some("Physics Camera Buffer"));
                SimpleBufferBind::new(ctx, camera, &layout, Some("Physics Camera Buffer Bind"))
            }
            Use::Active(camera) => camera,
            Use::Recycle(camera) | Use::Unused(camera) => {
                let data = viewport.as_raw();
                camera.buffer().update(ctx, data);
                camera
            }
        };
        self.camera = Use::Active(camera);
    }

    /// Update the instances of all moving objects
    fn update_instances(&mut self, ctx: &GpuContext, state: &SimulationState) {
        let rect_instances: Vec<_> = state
            .moving
            .iter()
            .map(|obj| match &obj.shape {
                MovingShape::Rectangle(rect) => rect.instance(),
            })
            .collect();

        let rect_instances = Instances::new(rect_instances);

        self.moving
            .update_instances(ctx, self.index_temp, &rect_instances);
    }
}
