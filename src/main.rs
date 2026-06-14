mod controls;
mod scene;

use std::sync::Arc;

use controls::Controls;
use iced_wgpu::graphics::{Shell, Viewport};
use iced_wgpu::{Engine, Renderer, wgpu};
use iced_winit::core::time::Instant;
use iced_winit::core::{Event, Font, Pixels, Size, Theme, mouse, renderer, window};
use iced_winit::runtime::user_interface::{self, UserInterface};
use iced_winit::winit::dpi::PhysicalPosition;
use iced_winit::winit::event::Modifiers;
use iced_winit::{Clipboard, conversion, futures, winit};
use scene::Scene;
use tracing::warn;
use winit::event::WindowEvent;
use winit::event_loop::{ControlFlow, EventLoop};
use winit::keyboard::ModifiersState;

pub fn main() -> Result<(), winit::error::EventLoopError> {
    tracing_subscriber::fmt::init();

    // Initialize winit
    let event_loop = EventLoop::new()?;

    let mut runner = Runner::Loading;
    event_loop.run_app(&mut runner)
}

struct Ready {
    window: Arc<winit::window::Window>,
    queue: wgpu::Queue,
    device: wgpu::Device,
    surface: wgpu::Surface<'static>,
    format: wgpu::TextureFormat,
    renderer: Renderer,
    controls: Controls,
    events: Vec<Event>,
    cursor: mouse::Cursor,
    cache: user_interface::Cache,
    clipboard: Clipboard,
    viewport: Viewport,
    modifiers: ModifiersState,
    resized: bool,
}

#[expect(clippy::large_enum_variant)]
enum Runner {
    Loading,
    Ready(Ready),
}

impl winit::application::ApplicationHandler for Runner {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let Self::Loading = self else {
            return;
        };

        *self = Self::Ready(Ready::new(event_loop));
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        use winit::event::KeyEvent;
        use winit::keyboard::{KeyCode, PhysicalKey};

        let Self::Ready(ready) = self else {
            return;
        };

        #[expect(clippy::wildcard_enum_match_arm)]
        match event {
            WindowEvent::RedrawRequested => ready.redraw(),
            WindowEvent::CursorMoved { position, .. } => ready.cursor_moved(position),
            WindowEvent::ModifiersChanged(modifiers) => ready.modifiers_changed(modifiers),
            WindowEvent::Resized(_) => ready.resized(),
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::KeyboardInput { ref event, .. } => {
                let KeyEvent { physical_key, .. } = event;
                match physical_key {
                    PhysicalKey::Code(KeyCode::KeyQ) if ready.modifiers.control_key() => {
                        event_loop.exit();
                    }
                    _ => (),
                }
            }
            _ => {}
        }

        // Map window event to iced event
        if let Some(event) =
            conversion::window_event(event, ready.window.scale_factor() as f32, ready.modifiers)
        {
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
                ready.controls.update(message);
            }

            // and request a redraw
            ready.window.request_redraw();
        }
    }
}

impl Ready {
    fn new(event_loop: &winit::event_loop::ActiveEventLoop) -> Self {
        let window = Arc::new(
            event_loop
                .create_window(winit::window::WindowAttributes::default())
                .expect("Create window"),
        );

        let physical_size = window.inner_size();
        let viewport = Viewport::with_physical_size(
            Size::new(physical_size.width, physical_size.height),
            window.scale_factor() as f32,
        );
        let clipboard = Clipboard::connect(Arc::clone(&window));

        let backends = wgpu::Backends::from_env().unwrap_or_default();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends,
            ..Default::default()
        });
        let surface = instance
            .create_surface(Arc::clone(&window))
            .expect("Create window surface");

        let (format, adapter, device, queue) = futures::futures::executor::block_on(async {
            let adapter =
                wgpu::util::initialize_adapter_from_env_or_default(&instance, Some(&surface))
                    .await
                    .expect("Create adapter");

            let adapter_features = adapter.features();
            let capabilities = surface.get_capabilities(&adapter);

            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: adapter_features & wgpu::Features::default(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    trace: wgpu::Trace::Off,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                })
                .await
                .expect("Request device");

            let format = capabilities
                .formats
                .iter()
                .copied()
                .find(wgpu::TextureFormat::is_srgb)
                .or_else(|| capabilities.formats.first().copied())
                .expect("Get preferred format");

            (format, adapter, device, queue)
        });

        surface.configure(
            &device,
            &wgpu::SurfaceConfiguration {
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                format,
                width: physical_size.width,
                height: physical_size.height,
                present_mode: wgpu::PresentMode::AutoVsync,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );

        // Initialize scene and GUI controls
        let scene = Scene::new(&device, format);
        let controls = Controls::new(scene);

        // Initialize iced

        let engine = Engine::new(
            &adapter,
            device.clone(),
            queue.clone(),
            format,
            None,
            Shell::headless(),
        );
        let renderer = Renderer::new(engine, Font::default(), Pixels::from(16));

        // You should change this if you want to render continuously
        event_loop.set_control_flow(ControlFlow::Wait);

        Self {
            window,
            device,
            queue,
            renderer,
            surface,
            format,
            controls,
            events: Vec::new(),
            cursor: mouse::Cursor::Unavailable,
            modifiers: ModifiersState::default(),
            cache: user_interface::Cache::new(),
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

        let frame = match self.surface.get_current_texture() {
            Ok(frame) => frame,
            Err(error) => {
                assert!(
                    error != wgpu::SurfaceError::OutOfMemory,
                    "Swapchain error, rendering cannot continue: {error}"
                );

                warn!("Error while drawing, try again next frame: {error}");
                // Try rendering again next frame.
                self.window.request_redraw();
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
        if let user_interface::State::Updated {
            mouse_interaction, ..
        } = state
        {
            // Update the mouse cursor
            if let Some(icon) = iced_winit::conversion::mouse_interaction(mouse_interaction) {
                self.window.set_cursor(icon);
                self.window.set_cursor_visible(true);
            } else {
                self.window.set_cursor_visible(false);
            }
        }

        // Draw the interface
        interface.draw(
            &mut self.renderer,
            &Theme::Dark,
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
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        self.controls.draw_wgpu(&view, &mut encoder);

        // Submit the scene
        self.queue.submit([encoder.finish()]);

        // Present the frame
        frame.present();
    }

    fn do_resize(&mut self) {
        let winit::dpi::PhysicalSize { width, height } = self.window.inner_size();

        self.viewport = Viewport::with_physical_size(
            Size::new(width, height),
            self.window.scale_factor() as f32,
        );

        self.surface.configure(
            &self.device,
            &wgpu::SurfaceConfiguration {
                format: self.format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                width,
                height,
                present_mode: wgpu::PresentMode::AutoNoVsync,
                alpha_mode: wgpu::CompositeAlphaMode::Auto,
                view_formats: vec![],
                desired_maximum_frame_latency: 2,
            },
        );
    }

    fn cursor_moved(&mut self, position: PhysicalPosition<f64>) {
        self.cursor = mouse::Cursor::Available(conversion::cursor_position(
            position,
            self.viewport.scale_factor(),
        ));
    }

    fn modifiers_changed(&mut self, modifiers: Modifiers) {
        self.modifiers = modifiers.state();
    }

    const fn resized(&mut self) {
        self.resized = true;
    }
}
