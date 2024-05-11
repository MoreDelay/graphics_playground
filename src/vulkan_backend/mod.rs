use std::collections::HashSet;
use std::ffi::{c_void, CStr};

use anyhow::{anyhow, Result};
use log::{debug, error, trace, warn};

use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_3::*;
use vulkanalia::vk::ExtDebugUtilsExtension;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

pub struct Vulkan {
    _entry: Entry,
    _instance: Instance,
    _debug_callback: Option<vk::DebugUtilsMessengerEXT>,
}

impl Vulkan {
    pub fn new(window: &winit::window::Window) -> Result<Vulkan> {
        debug!("Vulkan::new");
        unsafe {
            let loader = LibloadingLoader::new(LIBRARY)?;
            let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;

            let available_layers = entry
                .enumerate_instance_layer_properties()?
                .iter()
                .map(|l| {
                    debug!("Found layer: {}", l.layer_name);
                    l.layer_name
                })
                .collect::<HashSet<_>>();

            if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
                return Err(anyhow!("Validation layer requested but not supported."));
            }

            let layers = if VALIDATION_ENABLED {
                vec![VALIDATION_LAYER.as_ptr()]
            } else {
                Vec::new()
            };

            let application_info = vk::ApplicationInfo::builder()
                .application_name(b"Graphics Playground\0")
                .application_version(vk::make_version(0, 1, 0))
                .engine_name(b"No Engine\0")
                .engine_version(vk::make_version(1, 0, 0))
                .api_version(vk::make_version(1, 3, 280));

            let mut extensions = vulkanalia::window::get_required_instance_extensions(window)
                .iter()
                .map(|e| e.as_ptr())
                .collect::<Vec<_>>();

            if VALIDATION_ENABLED {
                extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
            }

            let flags = vk::InstanceCreateFlags::empty();

            let mut info = vk::InstanceCreateInfo::builder()
                .application_info(&application_info)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions)
                .flags(flags);

            // Make callback for all types of severities and all types of messages.
            let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
                .message_type(vk::DebugUtilsMessageTypeFlagsEXT::all())
                .user_callback(Some(debug_callback));

            if VALIDATION_ENABLED {
                info = info.push_next(&mut debug_info);
            }

            let instance = entry.create_instance(&info, None)?;

            let debug_callback = if VALIDATION_ENABLED {
                // Register callback with Vulkan instance for all other debug messages.
                Some(instance.create_debug_utils_messenger_ext(&debug_info, None)?)
            } else {
                None
            };

            return Ok(Self {
                _entry: entry,
                _instance: instance,
                _debug_callback: debug_callback,
            });
        }
    }
}

// Callback for Vulkan validation layer, so we specify extern "system"
// To see these messages printed out, set the environment variable RUST_LOG to the desired level.
// returns: should the Vulkan call be arborted? Used to test validation layer itself.
extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    message_type: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    match severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            error!("({:?}) {}", message_type, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            warn!("({:?}) {}", message_type, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            debug!("({:?}) {}", message_type, message)
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            trace!("({:?}) {}", message_type, message)
        }
        _ => error!("Unexpected severity"),
    }
    vk::FALSE
}
