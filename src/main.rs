use iced_winit::winit::error::EventLoopError;

mod app;
mod controls;
mod hello_triangle;
mod image;
mod instruments;

pub fn main() -> Result<(), EventLoopError> {
    tracing_subscriber::fmt::init();

    app::run()
}
