use graphics_playground::app::App;

use anyhow::Result;

fn main() -> Result<()> {
    pretty_env_logger::init();

    App::run()?;

    Ok(())
}
