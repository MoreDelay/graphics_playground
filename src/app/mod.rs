#![allow(dead_code, unused_variables)]

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

use anyhow::Result;
use log::info;

use crate::vulkan_backend::Vulkan;

#[derive(Default)]
pub struct App {
    debug: bool,
    window: Option<Window>,
    vulkan: Option<Vulkan>,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        info!("App::resumed");
        self.window = Some(
            event_loop
                .create_window(Window::default_attributes())
                .unwrap(),
        );

        if let None = self.vulkan {
            let Some(window) = &self.window else {
                unreachable!("Window set right above");
            };
            self.vulkan = Some(Vulkan::new(self.debug).unwrap());
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        info!("App::window_event");
        match event {
            WindowEvent::CloseRequested => {
                info!("CloseRequest");
                event_loop.exit();
            }
            WindowEvent::RedrawRequested => {
                info!("RedrawRequest");
                self.render().unwrap();
            }
            _ => {}
        }
    }
}

impl App {
    pub fn run(debug: bool) -> Result<()> {
        info!("App::run");

        let event_loop = EventLoop::new()?;
        event_loop.set_control_flow(ControlFlow::Poll);
        let mut app = App {
            debug,
            ..Default::default()
        };

        // Returns only on exit.
        event_loop.run_app(&mut app)?;

        Ok(())
    }

    fn render(self: &Self) -> Result<()> {
        info!("App::render");
        Ok(())
    }
}
