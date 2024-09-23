use graphics_playground::app::App;

use anyhow::Result;

fn main() -> Result<()> {
    env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .format_timestamp_secs()
        .parse_env("RUST_LOG")
        .init();

    App::run()?;
    Ok(())
}
