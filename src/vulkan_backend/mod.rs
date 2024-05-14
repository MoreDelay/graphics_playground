#![allow(dead_code, unused_variables)]

use std::collections::HashSet;
use std::ffi::{c_void, CStr};

use anyhow::{anyhow, Result};
use ash::vk::{
    DebugUtilsMessengerEXT, PhysicalDevice, PresentModeKHR, SurfaceCapabilitiesKHR,
    SurfaceFormatKHR, SurfaceKHR,
};
use log::{debug, error, trace, warn};

use ash::ext::debug_utils::Instance as DebugLoader;
use ash::khr::surface::Instance as SurfaceLoader;
use ash::{vk, Entry, Instance};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

pub struct Vulkan {
    entry: Entry,
    instance: Instance,
    debug_loader: DebugLoader,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    surface: vk::SurfaceKHR,
}

impl Vulkan {
    pub fn new(window: &winit::window::Window) -> Result<Vulkan> {
        debug!("Vulkan::new");
        let (entry, instance, debug_loader, debug_messenger) = Vulkan::create_instance(window)?;

        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
        }?;

        let surface_loader = SurfaceLoader::new(&entry, &instance);
        let (physical_device, surface_capabilities, surface_formats, surface_present_modes) =
            Vulkan::select_physical_device(&instance, &surface_loader, &surface)?;

        let msaa_samples = Vulkan::get_supported_msaa_sample_count(&instance, &physical_device);

        return Ok(Self {
            entry,
            instance,
            debug_loader,
            debug_messenger,
            surface,
        });
    }

    fn create_instance(
        window: &winit::window::Window,
    ) -> Result<(Entry, Instance, DebugLoader, Option<DebugUtilsMessengerEXT>)> {
        let entry = unsafe { Entry::load() }?;

        let props = unsafe { entry.enumerate_instance_layer_properties() }?;
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

        let instance = unsafe { entry.create_instance(&info, None) }?;

        let debug_loader = DebugLoader::new(&entry, &instance);

        let debug_messenger = if VALIDATION_ENABLED {
            Some(unsafe { debug_loader.create_debug_utils_messenger(&debug_info, None) }?)
        } else {
            None
        };

        Ok((entry, instance, debug_loader, debug_messenger))
    }

    fn select_physical_device(
        instance: &Instance,
        surface_loader: &SurfaceLoader,
        surface: &SurfaceKHR,
    ) -> Result<(
        vk::PhysicalDevice,
        SurfaceCapabilitiesKHR,
        Vec<SurfaceFormatKHR>,
        Vec<PresentModeKHR>,
    )> {
        for physical_device in unsafe { instance.enumerate_physical_devices() }? {
            let properties_device =
                unsafe { instance.get_physical_device_properties(physical_device) };
            if properties_device.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
                continue;
            }

            // Keep this here if I need to check for other features in the future.
            // let features = unsafe { instance.get_physical_device_features(physical_device) };
            // if features.geometry_shader != vk::TRUE {
            //     continue;
            // }
            // if features.sampler_anisotropy != vk::TRUE {
            //     continue;
            // }

            let properties_queues =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            // Graphics queues also support transfer operations as per spec.
            let queue_graphics = properties_queues
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
                .map(|i| i as u32);
            if let None = queue_graphics {
                continue;
            }

            let queue_present = (0..properties_queues.len())
                .position(|queue_index| unsafe {
                    surface_loader
                        .get_physical_device_surface_support(
                            physical_device,
                            queue_index as u32,
                            *surface,
                        )
                        .is_ok()
                })
                .map(|i| i as u32);
            if let None = queue_present {
                continue;
            }

            let extensions_available =
                unsafe { instance.enumerate_device_extension_properties(physical_device) }?
                    .iter()
                    .map(|e| {
                        e.extension_name
                            .iter()
                            .map(|v| *v as u8)
                            .collect::<Vec<_>>()
                    })
                    .collect::<HashSet<_>>();

            let extensions_needed = &[ash::khr::swapchain::NAME];
            // Make type system happy.
            let extensions_available = extensions_available
                .iter()
                .map(|e| CStr::from_bytes_until_nul(e).unwrap())
                .collect::<Vec<_>>();

            if !extensions_needed
                .iter()
                .all(|e| extensions_available.contains(e))
            {
                continue;
            }

            let Ok(swapchain_capabilities) = (unsafe {
                surface_loader.get_physical_device_surface_capabilities(physical_device, *surface)
            }) else {
                continue;
            };
            let Ok(swapchain_formats) = (unsafe {
                surface_loader.get_physical_device_surface_formats(physical_device, *surface)
            }) else {
                continue;
            };
            let Ok(swapchain_present_modes) = (unsafe {
                surface_loader.get_physical_device_surface_present_modes(physical_device, *surface)
            }) else {
                continue;
            };

            if swapchain_formats.is_empty() || swapchain_present_modes.is_empty() {
                continue;
            }

            return Ok((
                physical_device,
                swapchain_capabilities,
                swapchain_formats,
                swapchain_present_modes,
            ));
        }
        return Err(anyhow!("No suitable physical device found"));
    }

    fn get_supported_msaa_sample_count(
        instance: &Instance,
        physical_device: &PhysicalDevice,
    ) -> vk::SampleCountFlags {
        let properties_device =
            unsafe { instance.get_physical_device_properties(*physical_device) };
        let msaa_counts = properties_device.limits.framebuffer_color_sample_counts
            & properties_device.limits.framebuffer_depth_sample_counts;
        [
            vk::SampleCountFlags::TYPE_64,
            vk::SampleCountFlags::TYPE_32,
            vk::SampleCountFlags::TYPE_16,
            vk::SampleCountFlags::TYPE_8,
            vk::SampleCountFlags::TYPE_4,
            vk::SampleCountFlags::TYPE_2,
        ]
        .iter()
        .find(|c| msaa_counts.contains(**c))
        .cloned()
        .unwrap_or(vk::SampleCountFlags::TYPE_1)
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        debug!("Drop Vulkan");
        unsafe {
            if let Some(cb) = self.debug_messenger {
                self.debug_loader.destroy_debug_utils_messenger(cb, None);
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
