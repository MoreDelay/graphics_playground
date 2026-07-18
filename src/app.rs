use std::sync::Arc;
use std::time::Instant;

use iced::Event;
use iced::advanced::mouse::Cursor;
use iced::futures::executor::block_on;
use iced_graphics::{Shell, Viewport};
use iced_wgpu::core::SmolStr;
use iced_wgpu::{Engine, Renderer, wgpu};
use iced_winit::conversion::{cursor_position, window_event};
use iced_winit::core::{renderer, window};
use iced_winit::runtime::user_interface::{Cache, State, UserInterface};
use iced_winit::{Clipboard, winit};
use tracing::warn;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::error::EventLoopError;
use winit::event::{ElementState, Modifiers, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, ModifiersState};
use winit::window::WindowAttributes;

use crate::controls::{Controls, Message};
use crate::gpu::{GpuContext, TargetContext};

pub fn run_app() -> Result<(), EventLoopError> {
    // Initialize winit
    let event_loop = EventLoop::new()?;

    let mut runner = Runner::Loading;
    event_loop.run_app(&mut runner)
}

#[expect(clippy::large_enum_variant)]
enum Runner {
    Loading,
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
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        use winit::event::KeyEvent;

        let Self::Ready(ready) = self else {
            return;
        };

        #[expect(clippy::wildcard_enum_match_arm)]
        match event {
            WindowEvent::RedrawRequested => ready.redraw(),
            WindowEvent::CursorMoved { position, .. } => ready.cursor_moved(position),
            WindowEvent::MouseInput { state, button, .. } => ready.mouse_input(button, state),
            WindowEvent::MouseWheel { delta, .. } => ready.scrolled(delta),
            WindowEvent::ModifiersChanged(modifiers) => ready.modifiers_changed(modifiers),
            WindowEvent::Resized(_) => ready.resized(),
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Character(ref symbol),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match symbol.as_str() {
                "q" if ready.modifiers.control_key() => {
                    event_loop.exit();
                }
                _ => ready.key_pressed(symbol.clone()),
            },
            _ => {}
        }

        // Map window event to iced event
        #[expect(clippy::cast_possible_truncation)]
        let scale_factor = ready.target_ctx.window.scale_factor() as f32;
        if let Some(event) = window_event(event, scale_factor, ready.modifiers) {
            ready.events.push(event);
        }

        // If there are events pending
        if !ready.events.is_empty() {
            // We process them
            let mut interface = UserInterface::build(
                ready.controls.view(),
                ready.viewport.logical_size(),
                std::mem::take(&mut ready.cache),
                &mut ready.renderer,
            );

            let mut messages = Vec::new();

            let _ = interface.update(
                &ready.events,
                ready.cursor,
                &mut ready.renderer,
                &mut ready.clipboard,
                &mut messages,
            );

            ready.events.clear();
            ready.cache = interface.into_cache();

            // update our UI with any messages
            for message in messages {
                ready
                    .controls
                    .update(message, &ready.gpu_ctx, &ready.target_ctx, &ready.cursor);
            }

            // and request a redraw
            ready.target_ctx.window.request_redraw();
        }
    }
}

struct Ready {
    gpu_ctx: GpuContext,
    target_ctx: TargetContext,
    renderer: Renderer,
    controls: Controls,
    events: Vec<Event>,
    cursor: Cursor,
    dragging: DraggingState,
    cache: Cache,
    clipboard: Clipboard,
    viewport: Viewport,
    modifiers: ModifiersState,
    resized: bool,
}

impl Ready {
    fn new(event_loop: &ActiveEventLoop) -> Self {
        // Initialize window with winit
        let mut window = WindowAttributes::default();
        window.min_inner_size = Some(Controls::min_window_size().into());
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
                    label: None,
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
        #[expect(clippy::cast_possible_truncation)]
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
        Self {
            gpu_ctx,
            target_ctx,
            renderer,
            controls,
            events: Vec::new(),
            cursor: Cursor::Unavailable,
            dragging: DraggingState::default(),
            modifiers: ModifiersState::default(),
            cache: Cache::new(),
            clipboard,
            viewport,
            resized: false,
        }
    }

    fn redraw(&mut self) {
        if self.resized {
            self.do_resize();
            self.resized = false;
        }

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
                return;
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Draw iced first
        let mut interface = UserInterface::build(
            self.controls.view(),
            self.viewport.logical_size(),
            std::mem::take(&mut self.cache),
            &mut self.renderer,
        );

        let (state, _) = interface.update(
            &[Event::Window(
                window::Event::RedrawRequested(Instant::now()),
            )],
            self.cursor,
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
            self.cursor,
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
        let mut encoder = self
            .gpu_ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        let scale_factor = self.target_ctx.window.scale_factor();
        self.controls
            .draw_wgpu(&self.gpu_ctx, &view, &mut encoder, scale_factor);

        // Submit the scene
        self.gpu_ctx.queue.submit([encoder.finish()]);

        // Present the frame
        frame.present();
    }

    fn do_resize(&mut self) {
        let PhysicalSize { width, height } = self.target_ctx.window.inner_size();
        self.target_ctx.config.width = width;
        self.target_ctx.config.height = height;

        #[expect(clippy::cast_possible_truncation)]
        let scale_factor = self.target_ctx.window.scale_factor() as f32;
        self.viewport = Viewport::with_physical_size(iced::Size::new(width, height), scale_factor);

        self.target_ctx
            .surface
            .configure(&self.gpu_ctx.device, &self.target_ctx.config);
    }

    fn cursor_moved(&mut self, position: PhysicalPosition<f64>) {
        let after = cursor_position(position, self.viewport.scale_factor());
        let before = self.cursor;
        self.cursor = Cursor::Available(after);

        let Cursor::Available(before) = before else {
            self.dragging = DraggingState::Released;
            return;
        };

        if self.dragging == DraggingState::Dragging {
            let offset = after - before;
            let message = Message::Drag(offset);

            self.controls
                .update(message, &self.gpu_ctx, &self.target_ctx, &self.cursor);
        }
    }

    const fn mouse_input(&mut self, button: MouseButton, state: ElementState) {
        match (button, state) {
            (MouseButton::Left, ElementState::Pressed) => self.dragging = DraggingState::Dragging,
            (MouseButton::Left, ElementState::Released) => self.dragging = DraggingState::Released,
            _ => (),
        }
    }

    fn scrolled(&mut self, delta: MouseScrollDelta) {
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
        if let Some(message) = message {
            self.controls
                .update(message, &self.gpu_ctx, &self.target_ctx, &self.cursor);
        }
    }

    fn key_pressed(&mut self, key: SmolStr) {
        let message = Message::KeyPress(key);
        self.controls
            .update(message, &self.gpu_ctx, &self.target_ctx, &self.cursor);
    }

    fn modifiers_changed(&mut self, modifiers: Modifiers) {
        self.modifiers = modifiers.state();
    }

    const fn resized(&mut self) {
        self.resized = true;
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum DraggingState {
    #[default]
    Released,
    Dragging,
}
