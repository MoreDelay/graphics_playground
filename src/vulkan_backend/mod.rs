#![allow(dead_code, unused_variables)]

use std::collections::HashSet;
use std::ffi::{c_void, CStr};

use anyhow::{anyhow, Result};
use ash::vk::{
    DebugUtilsMessengerEXT, Extent2D, Image, PhysicalDevice, PresentModeKHR,
    SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR, SwapchainKHR,
};
use log::{debug, error, trace, warn};

use ash::ext::debug_utils::Instance as DebugLoader;
use ash::khr::surface::Instance as SurfaceLoader;
use ash::khr::swapchain::Device as SwapchainLoader;
use ash::{vk, Device, Entry, Instance};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

struct SwapchainContext {
    swapchain_loader: SwapchainLoader,
    swapchain: SwapchainKHR,
    swapchain_images: Vec<Image>,
    swapchain_format_used: SurfaceFormatKHR,
    swapchain_extent: Extent2D,
}

struct SwapchainSupport {
    capabilities: SurfaceCapabilitiesKHR,
    formats: Vec<SurfaceFormatKHR>,
    present_modes: Vec<PresentModeKHR>,
}

struct QueueIndices {
    graphics: u32,
    present: u32,
}

struct DeviceContext {
    device: Device,
    queue_graphics: vk::Queue,
    queue_present: vk::Queue,
}

pub struct Vulkan {
    entry: Entry,
    instance: Instance,
    debug_loader: DebugLoader,
    surface_loader: SurfaceLoader,
    debug_messenger: Option<vk::DebugUtilsMessengerEXT>,
    surface: vk::SurfaceKHR,
    physical_device: PhysicalDevice,

    queue_indices: QueueIndices,
    swapchain_support: SwapchainSupport,
    device_context: DeviceContext,
    swapchain_context: SwapchainContext,
}

impl Vulkan {
    pub fn new(window: &winit::window::Window) -> Result<Vulkan> {
        debug!("Vulkan::new");
        let (entry, instance, debug_loader, debug_messenger) = Vulkan::create_instance(window)?;

        let surface_loader = SurfaceLoader::new(&entry, &instance);
        let surface = unsafe {
            ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle()?.as_raw(),
                window.window_handle()?.as_raw(),
                None,
            )
        }?;

        let (physical_device, queue_indices, swapchain_support) =
            Vulkan::select_physical_device(&instance, &surface_loader, &surface)?;

        let msaa_samples = Vulkan::get_supported_msaa_sample_count(&instance, &physical_device);

        let device_context =
            Vulkan::create_logical_device(&instance, &physical_device, &queue_indices)?;

        let swapchain_context = Vulkan::create_swapchain(
            &instance,
            &device_context.device,
            &window,
            &surface,
            &swapchain_support,
            &queue_indices,
        )?;

        return Ok(Self {
            entry,
            instance,
            debug_loader,
            surface_loader,
            debug_messenger,
            surface,
            physical_device,
            queue_indices,
            swapchain_support,
            device_context,
            swapchain_context,
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
    ) -> Result<(vk::PhysicalDevice, QueueIndices, SwapchainSupport)> {
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
            let Some(queue_graphics) = queue_graphics else {
                continue;
            };

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
            let Some(queue_present) = queue_present else {
                continue;
            };

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

            let queue_indices = QueueIndices {
                graphics: queue_graphics,
                present: queue_present,
            };

            let swapchain_support = SwapchainSupport {
                capabilities: swapchain_capabilities,
                formats: swapchain_formats,
                present_modes: swapchain_present_modes,
            };
            return Ok((physical_device, queue_indices, swapchain_support));
        }
        return Err(anyhow!("No suitable physical device found"));
    }

    fn create_logical_device(
        instance: &Instance,
        physical_device: &PhysicalDevice,
        queue_indices: &QueueIndices,
    ) -> Result<DeviceContext> {
        let mut unique_indices = HashSet::new();
        unique_indices.insert(queue_indices.graphics);
        unique_indices.insert(queue_indices.present);

        let queue_priorities = &[1.0];
        let queue_infos = unique_indices
            .iter()
            .map(|i| {
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(*i)
                    .queue_priorities(queue_priorities)
            })
            .collect::<Vec<_>>();

        let extensions = vec![ash::khr::swapchain::NAME.as_ptr()];

        let features = vk::PhysicalDeviceFeatures::default()
            .sampler_anisotropy(false)
            .sample_rate_shading(false);

        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions)
            .enabled_features(&features);

        let device = unsafe { instance.create_device(*physical_device, &info, None) }?;

        let queue_graphics = unsafe { device.get_device_queue(queue_indices.graphics, 0) };
        let queue_present = unsafe { device.get_device_queue(queue_indices.present, 0) };

        let device_context = DeviceContext {
            device,
            queue_graphics,
            queue_present,
        };

        Ok(device_context)
    }

    fn create_swapchain(
        instance: &Instance,
        device: &Device,
        window: &winit::window::Window,
        surface: &SurfaceKHR,
        swapchain_support: &SwapchainSupport,
        queue_indices: &QueueIndices,
    ) -> Result<SwapchainContext> {
        let swapchain_format_used = swapchain_support
            .formats
            .iter()
            .find(|f| {
                f.format == vk::Format::R8G8B8A8_SRGB
                    && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .cloned()
            // If not found choose any format.
            .unwrap_or_else(|| swapchain_support.formats[0]);

        let swapchain_present_mode_used = swapchain_support
            .present_modes
            .iter()
            .find(|m| **m == vk::PresentModeKHR::MAILBOX) // triple buffering
            .cloned()
            .unwrap_or(vk::PresentModeKHR::FIFO); // guaranteed to be available

        let swapchain_extent = if swapchain_support.capabilities.current_extent.width != u32::MAX {
            swapchain_support.capabilities.current_extent
        } else {
            vk::Extent2D::default()
                .width(window.inner_size().width.clamp(
                    swapchain_support.capabilities.min_image_extent.width,
                    swapchain_support.capabilities.max_image_extent.width,
                ))
                .height(window.inner_size().height.clamp(
                    swapchain_support.capabilities.min_image_extent.height,
                    swapchain_support.capabilities.max_image_extent.height,
                ))
        };

        let mut image_count = swapchain_support.capabilities.min_image_count + 1;
        if swapchain_support.capabilities.max_image_count != 0
            && image_count > swapchain_support.capabilities.max_image_count
        {
            image_count = swapchain_support.capabilities.max_image_count;
        }

        // If queues are separate, use concurrent so we do not have to deal with ownership.
        let mut queue_family_indices = vec![];
        let image_sharing_mode = if queue_indices.graphics != queue_indices.present {
            queue_family_indices.push(queue_indices.graphics);
            queue_family_indices.push(queue_indices.present);
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(swapchain_format_used.format)
            .image_color_space(swapchain_format_used.color_space)
            .image_extent(swapchain_extent)
            .image_array_layers(1)
            .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(image_sharing_mode)
            .queue_family_indices(&queue_family_indices)
            .pre_transform(swapchain_support.capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(swapchain_present_mode_used)
            .clipped(true)
            .old_swapchain(vk::SwapchainKHR::null());

        let swapchain_loader = SwapchainLoader::new(instance, device);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&info, None) }?;
        let swapchain_images = unsafe { swapchain_loader.get_swapchain_images(swapchain) }?;

        let swapchain_context = SwapchainContext {
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_format_used,
            swapchain_extent,
        };

        Ok(swapchain_context)
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
            self.swapchain_context
                .swapchain_loader
                .destroy_swapchain(self.swapchain_context.swapchain, None)
        };

        unsafe { self.device_context.device.destroy_device(None) };

        unsafe { self.surface_loader.destroy_surface(self.surface, None) };

        if let Some(cb) = self.debug_messenger {
            unsafe { self.debug_loader.destroy_debug_utils_messenger(cb, None) };
        }

        unsafe { self.instance.destroy_instance(None) };
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
