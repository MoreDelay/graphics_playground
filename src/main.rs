use iced_winit::winit::error::EventLoopError;

mod app;
mod controls;
mod image;
mod instruments;
mod scene;

pub fn main() -> Result<(), EventLoopError> {
    tracing_subscriber::fmt::init();

    app::run_app()
}
