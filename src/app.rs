//! Contains the app's entry point and state struct

use std::sync::Arc;
use std::time::Instant;

use iced::Event;
use iced::futures::executor::block_on;
use iced_graphics::{Shell, Viewport};
use iced_wgpu::core::SmolStr;
use iced_wgpu::{Engine, Renderer, wgpu};
use iced_winit::conversion::window_event;
use iced_winit::core::{renderer, window};
use iced_winit::runtime::user_interface::{Cache, State, UserInterface};
use iced_winit::winit::event_loop::ControlFlow;
use iced_winit::{Clipboard, winit};
use nalgebra as na;
use tracing::warn;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::error::EventLoopError;
use winit::event::{ElementState, Modifiers, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::Key;
use winit::window::WindowAttributes;

use crate::controls::coords::Physical;
use crate::controls::{Controls, Message, Response, Updates};
use crate::instruments::{GpuContext, TargetContext};

/// The entry point to run the app
pub fn run() -> Result<(), EventLoopError> {
    // Initialize winit
    let event_loop = EventLoop::new()?;

    let mut runner = Runner::Loading;
    event_loop.run_app(&mut runner)
}

/// The app state
#[expect(clippy::large_enum_variant)]
enum Runner {
    /// No state yet, probably because we are still loading
    Loading,
    /// The window and render context has been initialized
    Ready(Ready),
}

impl winit::application::ApplicationHandler for Runner {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Self::Loading = self else {
            return;
        };

        *self = Self::Ready(Ready::new(event_loop));
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let Self::Ready(ready) = self {
            ready.window_event(event_loop, window_id, event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if let Self::Ready(ready) = self {
            ready.about_to_wait(event_loop);
        }
    }
}

impl Runner {}

/// Struct to hold all window and rendering state
struct Ready {
    // context objects
    /// Our gpu context
    gpu_ctx: GpuContext,
    /// Our gpu window target context
    target_ctx: TargetContext,

    // state of gui
    /// The controller of the app
    controls: Controls,
    /// Whether we need to update render state due to a resize
    resized: bool,

    // state of application
    /// The next time we should poll the controller
    next_poll: Option<Instant>,

    // objects used by iced but otherwise unused
    /// Iced renderer
    renderer: Renderer,
    /// Iced events
    events: Vec<Event>,
    /// Iced cache
    cache: Cache,
    /// Iced clipboard
    clipboard: Clipboard,
    /// Iced viewport
    viewport: Viewport,
}

impl Ready {
    /// Construct the whole app state from scratch
    fn new(event_loop: &ActiveEventLoop) -> Self {
        // Initialize window with winit
        let mut window = WindowAttributes::default();
        window.min_inner_size = Some(Controls::min_window_size().into());
        window.title = String::from("Graphics Playground");
        let window = Arc::new(event_loop.create_window(window).expect("Create window"));

        // Initialize wgpu
        let backends = wgpu::Backends::from_env().unwrap_or_default();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("Create window surface");

        let (format, adapter, device, queue) = block_on(async {
            let adapter =
                wgpu::util::initialize_adapter_from_env_or_default(&instance, Some(&surface))
                    .await
                    .expect("Create adapter");

            let required_features = adapter.features() & wgpu::Features::default();
            let required_features =
                required_features | wgpu::Features::ADDRESS_MODE_CLAMP_TO_BORDER;

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("Main Device"),
                    required_features,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                })
                .await
                .expect("Request device");

            let capabilities = surface.get_capabilities(&adapter);
            let format = capabilities
                .formats
                .iter()
                .copied()
                .find(wgpu::TextureFormat::is_srgb)
                .or_else(|| capabilities.formats.first().copied())
                .expect("Get preferred format");

            (format, adapter, device, queue)
        });

        let physical_size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: physical_size.width,
            height: physical_size.height,
            present_mode: wgpu::PresentMode::AutoNoVsync,
            alpha_mode: wgpu::CompositeAlphaMode::Auto,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        let gpu_ctx = GpuContext { device, queue };
        let target_ctx = TargetContext {
            window: Arc::clone(&window),
            surface,
            config: surface_config,
        };

        // Initialize scene and GUI controls
        let controls = Controls::new(&gpu_ctx, &target_ctx);

        // Initialize iced
        let scale_factor = window.scale_factor() as f32;
        let viewport = Viewport::with_physical_size(
            iced::Size::new(physical_size.width, physical_size.height),
            scale_factor,
        );
        let clipboard = Clipboard::connect(window);

        let engine = Engine::new(
            &adapter,
            gpu_ctx.device.clone(),
            gpu_ctx.queue.clone(),
            format,
            None,
            Shell::headless(),
        );
        let renderer = Renderer::new(engine, iced::Font::default(), iced::Pixels::from(16));

        // You should change this if you want to render continuously
        event_loop.set_control_flow(ControlFlow::Wait);

        let events = Vec::new();
        let cache = Cache::new();
        let resized = false;
        let next_poll = None;

        Self {
            gpu_ctx,
            target_ctx,
            controls,
            resized,
            next_poll,
            renderer,
            events,
            cache,
            clipboard,
            viewport,
        }
    }

    /// Handle winit window events
    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        use winit::event::KeyEvent;

        let now = Instant::now();
        if let Some(next) = self.next_poll
            && next <= now
        {
            let response = self.poll();
            self.handle_response(response, event_loop);
        }

        #[expect(clippy::wildcard_enum_match_arm)]
        let response = match event {
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::CursorMoved { position, .. } => self.cursor_moved(position),
            WindowEvent::MouseInput { state, button, .. } => self.mouse_input(button, state),
            WindowEvent::MouseWheel { delta, .. } => self.scrolled(delta),
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers_changed(modifiers),
            WindowEvent::Resized(_) => self.resized(),
            WindowEvent::CloseRequested => Response::Exit,
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref symbol),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => self.key_pressed(symbol.clone()),
            _ => Response::NoChange,
        };
        self.handle_response(response, event_loop);

        // Map window event to iced event
        let scale_factor = self.target_ctx.window.scale_factor() as f32;
        {
            if let Some(event) = window_event(event, scale_factor, self.controls.modifiers()) {
                self.events.push(event);
            }
        }

        // If there are events pending
        if !self.events.is_empty() {
            // We process them
            let mut interface = UserInterface::build(
                self.controls.view(scale_factor),
                self.viewport.logical_size(),
                std::mem::take(&mut self.cache),
                &mut self.renderer,
            );

            let mut messages = Vec::new();

            let _ = interface.update(
                &self.events,
                self.controls.cursor(),
                &mut self.renderer,
                &mut self.clipboard,
                &mut messages,
            );

            self.events.clear();
            self.cache = interface.into_cache();

            // update our UI with any messages
            for message in messages {
                let response = self
                    .controls
                    .update(&self.gpu_ctx, &self.target_ctx, message);
                self.handle_response(response, event_loop);
            }

            // and request a redraw
            self.target_ctx.window.request_redraw();
        }
    }

    /// Queue up custom redraw requests before waiting
    fn about_to_wait(&self, event_loop: &ActiveEventLoop) {
        let Some(next) = self.next_poll else {
            event_loop.set_control_flow(ControlFlow::Wait);
            return;
        };

        let now = Instant::now();
        if now < next {
            event_loop.set_control_flow(ControlFlow::WaitUntil(next));
        } else {
            self.target_ctx.window.request_redraw();
        }
    }

    /// Draw the app to the window
    fn redraw(&mut self) -> Response {
        let response = if self.resized {
            self.resized = false;
            self.reconfigure_surface()
        } else {
            Response::default()
        };

        let scale_factor = self.target_ctx.window.scale_factor() as f32;
        let frame = match self.target_ctx.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(error) => {
                assert!(
                    error != wgpu::SurfaceError::OutOfMemory,
                    "Swapchain error, rendering cannot continue: {error}"
                );

                warn!("Error while drawing, try again next frame: {error}");
                // Try rendering again next frame.
                self.target_ctx.window.request_redraw();
                return response;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Draw iced first
        let mut interface = UserInterface::build(
            self.controls.view(scale_factor),
            self.viewport.logical_size(),
            std::mem::take(&mut self.cache),
            &mut self.renderer,
        );

        let (state, _) = interface.update(
            &[Event::Window(
                window::Event::RedrawRequested(Instant::now()),
            )],
            self.controls.cursor(),
            &mut self.renderer,
            &mut self.clipboard,
            &mut Vec::new(),
        );

        // Update the mouse cursor
        if let State::Updated {
            mouse_interaction, ..
        } = state
        {
            // Update the mouse cursor
            if let Some(icon) = iced_winit::conversion::mouse_interaction(mouse_interaction) {
                self.target_ctx.window.set_cursor(icon);
                self.target_ctx.window.set_cursor_visible(true);
            } else {
                self.target_ctx.window.set_cursor_visible(false);
            }
        }

        // Draw the interface
        interface.draw(
            &mut self.renderer,
            &iced::Theme::Dark,
            &renderer::Style::default(),
            self.controls.cursor(),
        );
        self.cache = interface.into_cache();

        let bg_color = iced::Color {
            r: (3. / 255.0),
            g: (46. / 255.0),
            b: (99. / 255.0),
            a: 1.,
        };
        self.renderer.present(
            Some(bg_color),
            frame.texture.format(),
            &view,
            &self.viewport,
        );

        // Draw the scene with wgpu now.
        self.controls
            .draw_wgpu(&self.gpu_ctx, &self.target_ctx, &view);

        // Present the frame
        frame.present();

        response
    }

    /// Setup the render state to a new window size
    fn reconfigure_surface(&mut self) -> Response {
        let PhysicalSize { width, height } = self.target_ctx.window.inner_size();
        self.target_ctx.config.width = width;
        self.target_ctx.config.height = height;

        let scale_factor = self.target_ctx.window.scale_factor() as f32;
        self.viewport = Viewport::with_physical_size(iced::Size::new(width, height), scale_factor);

        let message = Message::SetScaleFactor(scale_factor);
        let response = self
            .controls
            .update(&self.gpu_ctx, &self.target_ctx, message);
        self.target_ctx
            .surface
            .configure(&self.gpu_ctx.device, &self.target_ctx.config);

        response
    }

    /// Handle when the cursor moved
    fn cursor_moved(&mut self, position: PhysicalPosition<f64>) -> Response {
        let PhysicalPosition { x, y } = position.cast();
        let position = Physical(na::Point2::new(x, y));
        let message = Message::CursorMoved(position);
        self.controls
            .update(&self.gpu_ctx, &self.target_ctx, message)
    }

    /// Handle when a mouse button was clicked
    fn mouse_input(&mut self, button: MouseButton, state: ElementState) -> Response {
        let message = Message::MouseInput { button, state };
        self.controls
            .update(&self.gpu_ctx, &self.target_ctx, message)
    }

    /// Handle when the mouse wheel was scrolled
    fn scrolled(&mut self, delta: MouseScrollDelta) -> Response {
        use std::cmp::Ordering;

        let cmp = match delta {
            MouseScrollDelta::LineDelta(_, delta) => delta.total_cmp(&0.),
            MouseScrollDelta::PixelDelta(PhysicalPosition { y, .. }) => y.total_cmp(&0.),
        };
        let message = match cmp {
            Ordering::Less => Some(Message::ScrollDown),
            Ordering::Equal => None,
            Ordering::Greater => Some(Message::ScrollUp),
        };
        let Some(message) = message else {
            return Response::NoChange;
        };
        self.controls
            .update(&self.gpu_ctx, &self.target_ctx, message)
    }

    /// Handle when a keyboard button was pressed
    fn key_pressed(&mut self, key: SmolStr) -> Response {
        let message = Message::KeyPress(key);
        self.controls
            .update(&self.gpu_ctx, &self.target_ctx, message)
    }

    /// Handle when a modifier key was pressed
    fn modifiers_changed(&mut self, modifiers: Modifiers) -> Response {
        let message = Message::ModifiersChanged(modifiers.state());
        self.controls
            .update(&self.gpu_ctx, &self.target_ctx, message)
    }

    /// Handle when a resize was requested
    const fn resized(&mut self) -> Response {
        self.resized = true;
        Response::NoChange
    }

    /// Do a poll to the controller
    fn poll(&mut self) -> Response {
        self.next_poll = None;
        let message = Message::Poll;
        self.controls
            .update(&self.gpu_ctx, &self.target_ctx, message)
    }

    /// Handle response from the controller
    #[expect(clippy::needless_pass_by_value, reason = "response is consumed here")]
    fn handle_response(&mut self, response: Response, event_loop: &ActiveEventLoop) {
        match response {
            Response::NoChange => (),
            Response::Exit => event_loop.exit(),
            Response::Updates(Updates::NextTime(instant)) => {
                self.next_poll = Some(instant);
                let now = Instant::now();
                if instant <= now {
                    self.target_ctx.window.request_redraw();
                } else {
                    event_loop.set_control_flow(ControlFlow::WaitUntil(instant));
                }
            }
            Response::Updates(Updates::OnEvent) => {
                self.next_poll = None;
                event_loop.set_control_flow(ControlFlow::Wait);
            }
        }
    }
}
