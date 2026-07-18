use iced_winit::winit::error::EventLoopError;

mod app;
mod controls;
mod gpu;
mod image;
mod scene;

pub fn main() -> Result<(), EventLoopError> {
    tracing_subscriber::fmt::init();

    app::run_app()
}
