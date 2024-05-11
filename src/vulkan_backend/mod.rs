use std::{collections::HashSet, sync::Arc};

use anyhow::{anyhow, Result};
use log::{debug, error, info, trace, warn};

use vulkano::{
    instance::{
        debug::{
            DebugUtilsMessageSeverity, DebugUtilsMessenger, DebugUtilsMessengerCallback,
            DebugUtilsMessengerCreateInfo,
        },
        Instance, InstanceCreateInfo, InstanceExtensions,
    },
    VulkanLibrary,
};

const VALIDATION_LAYER: &str = "VK_LAYER_KHRONOS_validation";

pub struct Vulkan {
    instance: Arc<Instance>,
    debug_messenger: DebugUtilsMessenger,
}

impl Vulkan {
    pub fn new(debug: bool) -> Result<Vulkan> {
        info!("Vulkan::new");
        let library = VulkanLibrary::new()?;

        library
            .layer_properties()?
            .for_each(|l| info!("Found layer: {}", l.name()));

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
            // vec![]
        } else {
            vec![]
        };

        let extensions = InstanceExtensions {
            ext_debug_utils: debug,
            // Can not currently automatically figure out the required extensions because this
            // requires a release after Vulkano 0.34 to be compatible with winit 0.30.
            // ..Surface::required_extensions(&event_loop)
            khr_surface: true,
            khr_wayland_surface: true,
            ..Default::default()
        };

        let mut debug_info;
        unsafe {
            let callback =
                DebugUtilsMessengerCallback::new(|severity, message_type, callback_data| {
                    let message = callback_data.message;

                    match severity {
                        DebugUtilsMessageSeverity::ERROR => {
                            error!("({:?}) {}", message_type, message)
                        }
                        DebugUtilsMessageSeverity::WARNING => {
                            warn!("({:?}) {}", message_type, message)
                        }
                        DebugUtilsMessageSeverity::INFO => {
                            info!("({:?}) {}", message_type, message)
                        }
                        DebugUtilsMessageSeverity::VERBOSE => {
                            trace!("({:?}) {}", message_type, message)
                        }
                        _ => error!("Unexpected severity"),
                    }
                });

            // Make a callback for all severity levels. By default this excludes verbose and info.
            debug_info = DebugUtilsMessengerCreateInfo::user_callback(callback);
            debug_info.message_severity = DebugUtilsMessageSeverity::VERBOSE
                | DebugUtilsMessageSeverity::INFO
                | DebugUtilsMessageSeverity::WARNING
                | DebugUtilsMessageSeverity::ERROR;
        }

        let info = InstanceCreateInfo {
            enabled_extensions: extensions,
            enabled_layers: layers,
            // This does not work in Vulkano 0.34 but is already fixed in main branch.
            // debug_utils_messengers: vec![debug_info.clone()],
            ..InstanceCreateInfo::application_from_cargo_toml()
        };

        let instance = Instance::new(library, info)?;

        // Keep the messenger alive until the end of the program or the callback will stop working.
        let debug_messenger = DebugUtilsMessenger::new(instance.clone(), debug_info)?;

        Ok(Self {
            instance,
            debug_messenger,
        })
    }
}
