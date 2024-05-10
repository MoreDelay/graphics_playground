use std::{collections::HashSet, sync::Arc};

use anyhow::{anyhow, Result};
use log::info;

use vulkano::{
    instance::{Instance, InstanceCreateInfo, InstanceExtensions},
    VulkanLibrary,
};

const VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";

pub struct Vulkan {
    instance: Arc<Instance>,
}

impl Vulkan {
    pub fn new(debug: bool) -> Result<Vulkan> {
        info!("Vulkan::new");
        let library = VulkanLibrary::new()?;

        library
            .layer_properties()?
            .for_each(|l| info!("Got layer: {}", l.name()));

        let layer_set = library
            .layer_properties()?
            .into_iter()
            .map(|l| String::from(l.name()))
            .collect::<HashSet<_>>();

        if debug && !layer_set.contains(VALIDATION_LAYER) {
            return Err(anyhow!("Validation layer requested but not supported."));
        }

        let layers = if debug {
            vec![String::from(VALIDATION_LAYER)]
        } else {
            vec![]
        };

        let extensions = InstanceExtensions {
            ext_debug_utils: true,
            // Can not currently automatically figure out the required extensions because this
            // requires a release after Vulkano 0.34 to be compatible with winit 0.30.
            // ..Surface::required_extensions(&event_loop)
            khr_surface: true,
            khr_wayland_surface: true,
            ..Default::default()
        };

        let info = InstanceCreateInfo {
            enabled_extensions: extensions,
            enabled_layers: layers,
            ..InstanceCreateInfo::application_from_cargo_toml()
        };

        let instance = Instance::new(library, info)?;

        Ok(Self { instance })
    }
}
