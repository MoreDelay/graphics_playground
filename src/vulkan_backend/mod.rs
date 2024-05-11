use std::collections::HashSet;
use std::ffi::{c_void, CStr};

use anyhow::{anyhow, Result};
use log::{debug, error, trace, warn};

use ash::{vk, Entry, Instance};
use winit::raw_window_handle::HasDisplayHandle;

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

pub struct Vulkan {
    _entry: Entry,
    instance: Instance,
    debug_utils: ash::ext::debug_utils::Instance,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
}

impl Vulkan {
    pub fn new(window: &winit::window::Window) -> Result<Vulkan> {
        debug!("Vulkan::new");
        unsafe {
            let entry = Entry::load()?;

            let props = entry.enumerate_instance_layer_properties()?;
            let available_layers = props
                .iter()
                .map(|l| {
                    debug!(
                        "Found layer: {}",
                        l.layer_name_as_c_str().unwrap().to_string_lossy()
                    );
                    l.layer_name_as_c_str().unwrap()
                })
                .collect::<HashSet<_>>();

            let validation_layer = std::ffi::CString::new("VK_LAYER_KHRONOS_validation").unwrap();

            if VALIDATION_ENABLED && !available_layers.contains(validation_layer.as_ref()) {
                return Err(anyhow!("Validation layer requested but not supported."));
            }

            let layers = if VALIDATION_ENABLED {
                vec![validation_layer.as_ptr()]
            } else {
                Vec::new()
            };

            let appname = std::ffi::CString::new("Graphics Playground").unwrap();
            let engname = std::ffi::CString::new("No Engine").unwrap();
            let application_info = vk::ApplicationInfo::default()
                .application_name(appname.as_ref())
                .application_version(vk::make_api_version(0, 0, 1, 0))
                .engine_name(engname.as_ref())
                .engine_version(vk::make_api_version(0, 0, 1, 0))
                .api_version(vk::make_api_version(0, 1, 3, 281));

            let mut extensions = Vec::from(ash_window::enumerate_required_extensions(
                window.display_handle()?.as_raw(),
            )?);

            if VALIDATION_ENABLED {
                extensions.push(ash::ext::debug_utils::NAME.as_ptr());
            }

            let flags = vk::InstanceCreateFlags::empty();

            let mut info = vk::InstanceCreateInfo::default()
                .application_info(&application_info)
                .enabled_layer_names(&layers)
                .enabled_extension_names(&extensions)
                .flags(flags);

            // Make callback for all types of severities and all types of messages.
            let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                        | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
                )
                .pfn_user_callback(Some(debug_callback));

            if VALIDATION_ENABLED {
                info = info.push_next(&mut debug_info);
            }

            let instance = entry.create_instance(&info, None)?;

            let debug_utils = ash::ext::debug_utils::Instance::new(&entry, &instance);

            let debug_messenger = if VALIDATION_ENABLED {
                Some(debug_utils.create_debug_utils_messenger(&debug_info, None)?)
            } else {
                None
            };

            return Ok(Self {
                _entry: entry,
                instance,
                debug_utils,
                debug_messenger,
            });
        }
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        debug!("Drop Vulkan");
        unsafe {
            if let Some(cb) = self.debug_messenger {
                self.debug_utils.destroy_debug_utils_messenger(cb, None);
            }

            self.instance.destroy_instance(None);
            debug!("Done");
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
    let message = unsafe { CStr::from_ptr(data.p_message) }.to_string_lossy();

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
