use iced_winit::winit::error::EventLoopError;

mod app;
mod controls;
mod hello_triangle;
mod image;
mod instruments;
mod model;
mod physics;
mod viewport;

pub fn main() -> Result<(), EventLoopError> {
    tracing_subscriber::fmt::init();

    app::run()
}
