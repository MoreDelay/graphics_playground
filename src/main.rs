use graphics_playground::app::App;

use anyhow::Result;

fn main() -> Result<()> {
    pretty_env_logger::init();

    let debug = true;
    App::run(debug)?;

    Ok(())
}
