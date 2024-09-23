use winit::{
    application::ApplicationHandler,
    event::{ElementState, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

use anyhow::{anyhow, Result};
use log::{info, trace};

use crate::vulkan_backend::Vulkan;

#[derive(Default)]
pub struct App {
    vulkan: Option<Vulkan>,
    // Window needs to stay after vulkan so that it outlifes destruction of vulkan surface. See
    // Rust drop order.
    window: Option<Window>,

    minimized: bool,
    resized: bool,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        info!("App::resumed");
        self.window = Some(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        if self.vulkan.is_none() {
            self.vulkan = Some(Vulkan::new(self.window.as_ref().unwrap()).unwrap());
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        trace!("App::window_event");
        match event {
            WindowEvent::CloseRequested => {
                info!("CloseRequest");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                trace!("RedrawRequest");
                self.render().unwrap();
            }
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    self.minimized = true;
                } else {
                    self.resized = true;
                    self.minimized = false;
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                info!("KeyboardInput");
                if event.state == ElementState::Pressed {
                    match event.physical_key {
                        PhysicalKey::Code(KeyCode::ArrowRight) => {
                            self.vulkan.as_mut().unwrap().add_model(1)
                        }
                        PhysicalKey::Code(KeyCode::ArrowLeft) => {
                            self.vulkan.as_mut().unwrap().sub_model(1)
                        }
                        _ => (),
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let _ = event_loop;
        match self.window.as_ref() {
            Some(window) => window.request_redraw(),
            None => todo!(),
        }
    }
}

impl App {
    pub fn run() -> Result<()> {
        info!("App::run");

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = App::default();

        event_loop.run_app(&mut app)?;

        Ok(())
    }

    fn render(self: &mut Self) -> Result<()> {
        trace!("App::render");
        let window = self
            .window
            .as_ref()
            .ok_or(anyhow!("No window to render to"))?;

        match self.vulkan.as_mut() {
            Some(vulkan) => vulkan.render(window),
            None => Err(anyhow!("Vulkan not initialized")),
        }
    }
}
