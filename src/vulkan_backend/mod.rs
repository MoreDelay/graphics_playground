#![allow(dead_code, unused_variables)]

use std::collections::HashSet;
use std::ffi::{c_void, CStr};
use std::mem::size_of;

use anyhow::{anyhow, Result};
use ash::vk::{
    CommandBuffer, CommandPool, DebugUtilsMessengerEXT, DescriptorSetLayout, DeviceMemory,
    Extent2D, Fence, Format, Framebuffer, Handle, Image, ImageView, PhysicalDevice, Pipeline,
    PipelineLayout, PresentModeKHR, RenderPass, SampleCountFlags, Semaphore,
    SurfaceCapabilitiesKHR, SurfaceFormatKHR, SurfaceKHR, SwapchainKHR,
};
use log::{debug, error, trace, warn};

use ash::ext::debug_utils::Instance as DebugLoader;
use ash::khr::surface::Instance as SurfaceLoader;
use ash::khr::swapchain::Device as SwapchainLoader;
use ash::{vk, Device, Entry, Instance};
use winit::raw_window_handle::{HasDisplayHandle, HasWindowHandle};

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);

type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;
type Vec4 = cgmath::Vector4<f32>;

type Mat4 = cgmath::Matrix4<f32>;

struct Vertex {
    position: Vec3,
    color: Vec3,
}

struct VertexBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

struct IndexBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
}

struct RenderContext {
    next_frame: usize,
}

struct SyncObjects {
    image_available_semaphores: Vec<Semaphore>,
    render_finished_semaphores: Vec<Semaphore>,
    in_flight_fences: Vec<Fence>,
    images_in_flight: Vec<Fence>,
}

struct CommandContext {
    command_pool: CommandPool,
    command_buffer: CommandBuffer,
    command_buffer_per_image: Vec<CommandBuffer>,
}

struct RenderObjects {
    color_image_view: ImageView,
    depth_image_view: ImageView,
    color_image_context: ImageContext,
    depth_image_context: ImageContext,
}

struct RenderPassContext {
    render_pass: RenderPass,
    depth_format: Format,
}

struct ImageContext {
    image: Image,
    image_memory: DeviceMemory,
}

struct PipelineContext {
    graphics_pipeline: Pipeline,
    descriptor_set_layout: DescriptorSetLayout,
    pipeline_layout: PipelineLayout,
}

struct SwapchainContext {
    swapchain_loader: SwapchainLoader,
    swapchain: SwapchainKHR,
    swapchain_images: Vec<Image>,
    swapchain_image_views: Vec<ImageView>,
    swapchain_format: SurfaceFormatKHR,
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

struct PhysicalDeviceContext {
    physical_device: PhysicalDevice,
    queue_index_graphics: u32,
    queue_index_present: u32,
    msaa_samples: SampleCountFlags,
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
    render_objects: RenderObjects,
    sync_objects: SyncObjects,
    framebuffers: Vec<Framebuffer>,

    physical_device_context: PhysicalDeviceContext,
    swapchain_support: SwapchainSupport,
    device_context: DeviceContext,
    swapchain_context: SwapchainContext,
    render_pass_context: RenderPassContext,
    pipeline_context: PipelineContext,
    command_context: CommandContext,
    render_context: RenderContext,
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

        let (physical_device_context, swapchain_support) =
            Vulkan::select_physical_device(&instance, &surface_loader, &surface)?;

        let device_context = Vulkan::create_logical_device(&instance, &physical_device_context)?;

        let swapchain_context = Vulkan::create_swapchain(
            &instance,
            &device_context.device,
            &window,
            &surface,
            &swapchain_support,
            &physical_device_context,
        )?;

        let render_pass_context = Vulkan::create_render_pass(
            &instance,
            &device_context.device,
            &physical_device_context,
            &swapchain_context,
        )?;

        let pipeline_context = Vulkan::create_pipeline(
            &device_context.device,
            &render_pass_context,
            &physical_device_context,
            &swapchain_context,
        )?;

        let render_objects = Vulkan::create_render_objects(
            &instance,
            &device_context.device,
            &physical_device_context,
            &swapchain_context,
            &render_pass_context,
        )?;

        let framebuffers = Vulkan::create_frame_buffers(
            &device_context.device,
            &swapchain_context,
            &render_pass_context,
            &render_objects,
        )?;

        let command_context = Vulkan::create_command_buffers(
            &device_context.device,
            &physical_device_context,
            &swapchain_context,
            &framebuffers,
        )?;

        let sync_objects =
            Vulkan::create_sync_objects(&device_context.device, &swapchain_context, 2)?;

        let render_context = RenderContext { next_frame: 0 };

        return Ok(Self {
            entry,
            instance,
            debug_loader,
            surface_loader,
            debug_messenger,
            surface,
            physical_device_context,
            swapchain_support,
            device_context,
            swapchain_context,
            render_pass_context,
            pipeline_context,
            command_context,
            render_objects,
            framebuffers,
            sync_objects,
            render_context,
        });
    }

    pub fn render(self: &mut Self, window: &winit::window::Window) -> Result<()> {
        let in_flight_fence = self.sync_objects.in_flight_fences[self.render_context.next_frame];
        unsafe {
            self.device_context
                .device
                .wait_for_fences(&[in_flight_fence], true, u64::MAX)
        }?;

        let image_index = unsafe {
            self.swapchain_context.swapchain_loader.acquire_next_image(
                self.swapchain_context.swapchain,
                u64::MAX,
                self.sync_objects.image_available_semaphores[self.render_context.next_frame],
                vk::Fence::null(),
            )
        };

        let image_index = match image_index {
            Ok((image_index, _)) => image_index as usize,
            // Err(vk::Result::ERROR_OUT_OF_DATE_KHR) =>
            Err(e) => return Err(anyhow!(e)),
        };

        let image_in_flight = self.sync_objects.images_in_flight[image_index];
        if !image_in_flight.is_null() {
            unsafe {
                self.device_context
                    .device
                    .wait_for_fences(&[image_in_flight], true, u64::MAX)
            }?;
        }
        self.sync_objects.images_in_flight[image_index] = in_flight_fence;

        Vulkan::update_command_buffer(
            &self.device_context.device,
            &mut self.command_context,
            &self.swapchain_context,
            &self.pipeline_context,
            &self.render_pass_context,
            &self.framebuffers,
            image_index,
        )?;

        let wait_semaphores =
            &[self.sync_objects.image_available_semaphores[self.render_context.next_frame]];
        let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let command_buffers = &[self.command_context.command_buffer_per_image[image_index]];
        let signal_semaphores =
            &[self.sync_objects.render_finished_semaphores[self.render_context.next_frame]];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_stages)
            .command_buffers(command_buffers)
            .signal_semaphores(signal_semaphores);

        unsafe {
            self.device_context
                .device
                .reset_fences(&[self.sync_objects.in_flight_fences[self.render_context.next_frame]])
        }?;

        unsafe {
            self.device_context.device.queue_submit(
                self.device_context.queue_graphics,
                &[submit_info],
                self.sync_objects.in_flight_fences[self.render_context.next_frame],
            )
        }?;

        let swapchains = &[self.swapchain_context.swapchain];
        let image_indices = &[image_index as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(signal_semaphores)
            .swapchains(swapchains)
            .image_indices(image_indices);

        let result = unsafe {
            self.swapchain_context
                .swapchain_loader
                .queue_present(self.device_context.queue_present, &present_info)
        };
        let changed = result == Err(vk::Result::SUBOPTIMAL_KHR)
            || result == Err(vk::Result::ERROR_OUT_OF_DATE_KHR);

        // if changed  || self.resized {
        //
        // }

        self.render_context.next_frame = (self.render_context.next_frame + 1) % 2;
        Ok(())
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
    ) -> Result<(PhysicalDeviceContext, SwapchainSupport)> {
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

            let msaa_counts = properties_device.limits.framebuffer_color_sample_counts
                & properties_device.limits.framebuffer_depth_sample_counts;
            let msaa_samples = [
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
            .unwrap_or(vk::SampleCountFlags::TYPE_1);

            let physical_device_context = PhysicalDeviceContext {
                physical_device,
                queue_index_graphics: queue_graphics,
                queue_index_present: queue_present,
                msaa_samples,
            };

            let swapchain_support = SwapchainSupport {
                capabilities: swapchain_capabilities,
                formats: swapchain_formats,
                present_modes: swapchain_present_modes,
            };
            return Ok((physical_device_context, swapchain_support));
        }
        return Err(anyhow!("No suitable physical device found"));
    }

    fn create_logical_device(
        instance: &Instance,
        physical_device_context: &PhysicalDeviceContext,
    ) -> Result<DeviceContext> {
        let mut unique_indices = HashSet::new();
        unique_indices.insert(physical_device_context.queue_index_graphics);
        unique_indices.insert(physical_device_context.queue_index_present);

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
            .sample_rate_shading(true);

        let info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions)
            .enabled_features(&features);

        let device = unsafe {
            instance.create_device(physical_device_context.physical_device, &info, None)
        }?;

        let queue_graphics =
            unsafe { device.get_device_queue(physical_device_context.queue_index_graphics, 0) };
        let queue_present =
            unsafe { device.get_device_queue(physical_device_context.queue_index_present, 0) };

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
        physical_device_context: &PhysicalDeviceContext,
    ) -> Result<SwapchainContext> {
        let swapchain_format = swapchain_support
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
        let image_sharing_mode = if physical_device_context.queue_index_graphics
            != physical_device_context.queue_index_present
        {
            queue_family_indices.push(physical_device_context.queue_index_graphics);
            queue_family_indices.push(physical_device_context.queue_index_present);
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let info = vk::SwapchainCreateInfoKHR::default()
            .surface(*surface)
            .min_image_count(image_count)
            .image_format(swapchain_format.format)
            .image_color_space(swapchain_format.color_space)
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

        let swapchain_image_views = swapchain_images
            .iter()
            .map(|i| {
                Vulkan::create_image_view(
                    device,
                    i,
                    &swapchain_format.format,
                    vk::ImageAspectFlags::COLOR,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();

        let swapchain_context = SwapchainContext {
            swapchain_loader,
            swapchain,
            swapchain_images,
            swapchain_image_views,
            swapchain_format,
            swapchain_extent,
        };

        Ok(swapchain_context)
    }

    fn create_render_pass(
        instance: &Instance,
        device: &Device,
        physical_device_context: &PhysicalDeviceContext,
        swapchain_context: &SwapchainContext,
    ) -> Result<RenderPassContext> {
        let color_attachment = vk::AttachmentDescription::default()
            .format(swapchain_context.swapchain_format.format)
            .samples(physical_device_context.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_format_candidates = &[
            vk::Format::D32_SFLOAT,
            vk::Format::D32_SFLOAT_S8_UINT,
            vk::Format::D24_UNORM_S8_UINT,
        ];

        let depth_format = depth_format_candidates
            .iter()
            .find(|f| {
                let properties = unsafe {
                    instance.get_physical_device_format_properties(
                        physical_device_context.physical_device,
                        **f,
                    )
                };
                properties
                    .optimal_tiling_features
                    .contains(vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT)
            })
            .cloned()
            .unwrap();

        let depth_stencil_attachment = vk::AttachmentDescription::default()
            .format(depth_format)
            .samples(physical_device_context.msaa_samples)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_resolve_attachment = vk::AttachmentDescription::default()
            .format(swapchain_context.swapchain_format.format)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::DONT_CARE)
            .store_op(vk::AttachmentStoreOp::DONT_CARE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

        let color_attachment_ref = vk::AttachmentReference::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let depth_stencil_attachment_ref = vk::AttachmentReference::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

        let color_resolve_attachment_ref = vk::AttachmentReference::default()
            .attachment(2)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

        let color_attachments = &[color_attachment_ref];
        let resolve_attachments = &[color_resolve_attachment_ref];
        let subpass = vk::SubpassDescription::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(color_attachments)
            .depth_stencil_attachment(&depth_stencil_attachment_ref)
            .resolve_attachments(resolve_attachments);

        let dependency = vk::SubpassDependency::default()
            .src_subpass(vk::SUBPASS_EXTERNAL)
            .dst_subpass(0)
            .src_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .src_access_mask(vk::AccessFlags::empty())
            .dst_stage_mask(
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            )
            .dst_access_mask(
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            );

        let attachments = &[
            color_attachment,
            depth_stencil_attachment,
            color_resolve_attachment,
        ];
        let subpasses = &[subpass];
        let dependencies = &[dependency];
        let info = vk::RenderPassCreateInfo::default()
            .attachments(attachments)
            .subpasses(subpasses)
            .dependencies(dependencies);

        let render_pass = unsafe { device.create_render_pass(&info, None) }?;

        let render_pass_context = RenderPassContext {
            render_pass,
            depth_format,
        };

        Ok(render_pass_context)
    }

    fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
        let mut cursor = std::io::Cursor::new(bytecode);
        let shader_code = ash::util::read_spv(&mut cursor)?;

        let info = vk::ShaderModuleCreateInfo::default().code(&shader_code);
        let module = unsafe { device.create_shader_module(&info, None) }?;

        Ok(module)
    }

    fn create_pipeline(
        device: &Device,
        render_pass_context: &RenderPassContext,
        physical_device_context: &PhysicalDeviceContext,
        swapchain_context: &SwapchainContext,
    ) -> Result<PipelineContext> {
        // Todo: compile on runtime
        let vert = include_bytes!("../../target/shaders/vert.spv");
        let frag = include_bytes!("../../target/shaders/frag.spv");

        let shader_module_vert = Vulkan::create_shader_module(device, vert)?;
        let shader_module_frag = Vulkan::create_shader_module(device, frag)?;

        let stage_vert = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader_module_vert)
            .name(CStr::from_bytes_with_nul(b"main\0")?);

        let stage_frag = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader_module_frag)
            .name(CStr::from_bytes_with_nul(b"main\0")?);

        let vertex_binding_descriptions = &[vk::VertexInputBindingDescription::default()
            .binding(0)
            .stride(size_of::<Vertex>() as u32)
            .input_rate(vk::VertexInputRate::VERTEX)];

        let vertex_attribute_descriptions = &[
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(0)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(0),
            vk::VertexInputAttributeDescription::default()
                .binding(0)
                .location(1)
                .format(vk::Format::R32G32B32_SFLOAT)
                .offset(size_of::<Vec3>() as u32),
        ];

        let vertex_input_state_info = vk::PipelineVertexInputStateCreateInfo::default();
        // .vertex_binding_descriptions(vertex_binding_descriptions)
        // .vertex_attribute_descriptions(vertex_attribute_descriptions);

        let vertex_input_assembly_state_info = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
            .primitive_restart_enable(false);

        let viewports = &[vk::Viewport::default()
            .x(0.0)
            .y(0.0)
            .width(swapchain_context.swapchain_extent.width as f32)
            .height(swapchain_context.swapchain_extent.height as f32)
            .min_depth(0.0)
            .max_depth(1.0)];

        let scissors = &[vk::Rect2D::default()
            .offset(vk::Offset2D { x: 0, y: 0 })
            .extent(swapchain_context.swapchain_extent)];

        let viewport_state_info = vk::PipelineViewportStateCreateInfo::default()
            .viewports(viewports)
            .scissors(scissors);

        let rasterization_state_info = vk::PipelineRasterizationStateCreateInfo::default()
            .depth_clamp_enable(false)
            .rasterizer_discard_enable(false)
            .polygon_mode(vk::PolygonMode::FILL)
            .line_width(1.0)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
            .depth_bias_enable(false);

        let multisample_state_info = vk::PipelineMultisampleStateCreateInfo::default()
            .sample_shading_enable(true)
            .min_sample_shading(0.2)
            .rasterization_samples(physical_device_context.msaa_samples);

        let color_blend_attachments = &[vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)
            .blend_enable(true)
            .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
            .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
            .color_blend_op(vk::BlendOp::ADD)
            .src_alpha_blend_factor(vk::BlendFactor::ONE)
            .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
            .alpha_blend_op(vk::BlendOp::ADD)];

        let color_blend_state_info = vk::PipelineColorBlendStateCreateInfo::default()
            .logic_op_enable(false)
            .logic_op(vk::LogicOp::COPY)
            .attachments(color_blend_attachments)
            .blend_constants([0.0, 0.0, 0.0, 0.0]);

        let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::LINE_WIDTH];
        let dynamic_state_info =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(dynamic_states);

        let push_constant_range_vert = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .offset(0)
            .size(size_of::<Mat4>() as u32);

        // let push_constant_range_frag = vk::PushConstantRange::default()
        //     .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        //     .offset(size_of::<Mat4>() as u32)
        //     .size(size_of::<f32>() as u32);

        let ubo_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::VERTEX);

        let sampler_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::FRAGMENT);

        let bindings = &[ubo_binding, sampler_binding];
        let descriptor_set_layout_info =
            vk::DescriptorSetLayoutCreateInfo::default().bindings(bindings);

        let descriptor_set_layout =
            unsafe { device.create_descriptor_set_layout(&descriptor_set_layout_info, None) }?;

        let set_layouts = &[descriptor_set_layout];
        let push_constant_ranges = &[push_constant_range_vert];
        let layout_info = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(set_layouts)
            .push_constant_ranges(push_constant_ranges);

        let pipeline_layout = unsafe { device.create_pipeline_layout(&layout_info, None) }?;

        let depth_stencil_state_info = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::LESS)
            .depth_bounds_test_enable(false)
            .min_depth_bounds(0.0)
            .max_depth_bounds(1.0)
            .stencil_test_enable(false)
            .front(vk::StencilOpState::default())
            .back(vk::StencilOpState::default());

        let stages = &[stage_vert, stage_frag];
        let graphics_pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(stages)
            .vertex_input_state(&vertex_input_state_info)
            .input_assembly_state(&vertex_input_assembly_state_info)
            .viewport_state(&viewport_state_info)
            .rasterization_state(&rasterization_state_info)
            .multisample_state(&multisample_state_info)
            .depth_stencil_state(&depth_stencil_state_info)
            .color_blend_state(&color_blend_state_info)
            .layout(pipeline_layout)
            .render_pass(render_pass_context.render_pass)
            .subpass(0)
            .base_pipeline_handle(vk::Pipeline::null())
            .base_pipeline_index(-1);

        let Ok(graphics_pipelines) = (unsafe {
            device.create_graphics_pipelines(
                vk::PipelineCache::null(),
                &[graphics_pipeline_info],
                None,
            )
        }) else {
            return Err(anyhow!("Pipeline could not be created"));
        };

        let graphics_pipeline = graphics_pipelines[0];

        unsafe { device.destroy_shader_module(shader_module_frag, None) };
        unsafe { device.destroy_shader_module(shader_module_vert, None) };

        let pipeline_context = PipelineContext {
            graphics_pipeline,
            descriptor_set_layout,
            pipeline_layout,
        };

        Ok(pipeline_context)
    }

    fn create_image(
        instance: &Instance,
        device: &Device,
        physical_device_context: &PhysicalDeviceContext,
        swapchain_context: &SwapchainContext,
        format: &Format,
        usage: vk::ImageUsageFlags,
    ) -> Result<ImageContext> {
        let info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .extent(vk::Extent3D {
                width: swapchain_context.swapchain_extent.width,
                height: swapchain_context.swapchain_extent.height,
                depth: 1,
            })
            .mip_levels(1)
            .array_layers(1)
            .format(*format)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE)
            .samples(physical_device_context.msaa_samples)
            .flags(vk::ImageCreateFlags::empty());
        let image = unsafe { device.create_image(&info, None) }?;

        let requirements = unsafe { device.get_image_memory_requirements(image) };
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device_context.physical_device)
        };
        let memory_type_index = (0..memory_properties.memory_type_count)
            .find(|i| {
                let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
                let mem_type = memory_properties.memory_types[*i as usize];
                suitable
                    && mem_type
                        .property_flags
                        .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
            })
            .ok_or_else(|| anyhow!("No suitable memory type"))?;

        let mem_alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index);
        let image_memory = unsafe { device.allocate_memory(&mem_alloc_info, None) }?;

        unsafe { device.bind_image_memory(image, image_memory, 0) }?;

        let image_context = ImageContext {
            image,
            image_memory,
        };

        Ok(image_context)
    }

    fn create_image_view(
        device: &Device,
        image: &Image,
        format: &vk::Format,
        aspect: vk::ImageAspectFlags,
    ) -> Result<ImageView> {
        let components = vk::ComponentMapping::default()
            .r(vk::ComponentSwizzle::IDENTITY)
            .g(vk::ComponentSwizzle::IDENTITY)
            .b(vk::ComponentSwizzle::IDENTITY)
            .a(vk::ComponentSwizzle::IDENTITY);
        let subresource_range = vk::ImageSubresourceRange::default()
            .aspect_mask(aspect)
            .base_mip_level(0)
            .level_count(1)
            .base_array_layer(0)
            .layer_count(1);

        let info = vk::ImageViewCreateInfo::default()
            .image(*image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(*format)
            .components(components)
            .subresource_range(subresource_range);

        let image_view = unsafe { device.create_image_view(&info, None) }?;

        Ok(image_view)
    }

    fn create_frame_buffers(
        device: &Device,
        swapchain_context: &SwapchainContext,
        render_pass_context: &RenderPassContext,
        render_objects: &RenderObjects,
    ) -> Result<Vec<Framebuffer>> {
        let framebuffers = swapchain_context
            .swapchain_image_views
            .iter()
            .map(|i| {
                let attachments = &[
                    render_objects.color_image_view,
                    render_objects.depth_image_view,
                    *i,
                ];
                let info = vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass_context.render_pass)
                    .attachments(attachments)
                    .width(swapchain_context.swapchain_extent.width)
                    .height(swapchain_context.swapchain_extent.height)
                    .layers(1);

                unsafe { device.create_framebuffer(&info, None) }
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(framebuffers)
    }

    fn create_command_pool(device: &Device, queue_family: u32) -> Result<CommandPool> {
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::empty()) // use transient with push constants
            .queue_family_index(queue_family);

        let command_pool = unsafe { device.create_command_pool(&info, None) }?;

        Ok(command_pool)
    }

    fn create_command_buffers(
        device: &Device,
        physical_device_context: &PhysicalDeviceContext,
        swapchain_context: &SwapchainContext,
        framebuffers: &Vec<Framebuffer>,
    ) -> Result<CommandContext> {
        let command_pool =
            Vulkan::create_command_pool(device, physical_device_context.queue_index_graphics)?;

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool) // TODO command pool per image
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info) }?[0];

        let n_images = swapchain_context.swapchain_images.len();
        let mut command_buffers = Vec::with_capacity(n_images);
        for image_index in 0..n_images {
            let allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool) // TODO command pool per image
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);

            let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info) }?[0];
            command_buffers.push(command_buffer);
        }

        let command_context = CommandContext {
            command_pool,
            command_buffer,
            command_buffer_per_image: command_buffers,
        };

        Ok(command_context)
    }

    fn create_render_objects(
        instance: &Instance,
        device: &Device,
        physical_device_context: &PhysicalDeviceContext,
        swapchain_context: &SwapchainContext,
        render_pass_context: &RenderPassContext,
    ) -> Result<RenderObjects> {
        let color_image_context = Vulkan::create_image(
            instance,
            device,
            physical_device_context,
            swapchain_context,
            &swapchain_context.swapchain_format.format,
            vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        )?;
        let color_image_view = Vulkan::create_image_view(
            device,
            &color_image_context.image,
            &swapchain_context.swapchain_format.format,
            vk::ImageAspectFlags::COLOR,
        )?;

        let depth_image_context = Vulkan::create_image(
            instance,
            device,
            physical_device_context,
            swapchain_context,
            &render_pass_context.depth_format,
            vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        )?;
        let depth_image_view = Vulkan::create_image_view(
            device,
            &depth_image_context.image,
            &render_pass_context.depth_format,
            vk::ImageAspectFlags::DEPTH,
        )?;

        let render_objects = RenderObjects {
            color_image_view,
            depth_image_view,
            color_image_context,
            depth_image_context,
        };

        Ok(render_objects)
    }

    fn create_sync_objects(
        device: &Device,
        swapchain_context: &SwapchainContext,
        max_frames_in_flight: usize,
    ) -> Result<SyncObjects> {
        let semaphore_info = vk::SemaphoreCreateInfo::default();
        let fence_info = vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

        let mut image_available_semaphores = Vec::new();
        let mut render_finished_semaphores = Vec::new();
        let mut in_flight_fences = Vec::new();
        let mut images_in_flight = Vec::new();

        for _ in 0..max_frames_in_flight {
            image_available_semaphores
                .push(unsafe { device.create_semaphore(&semaphore_info, None) }?);
            render_finished_semaphores
                .push(unsafe { device.create_semaphore(&semaphore_info, None) }?);
            in_flight_fences.push(unsafe { device.create_fence(&fence_info, None) }?);
        }

        swapchain_context
            .swapchain_images
            .iter()
            .for_each(|_| images_in_flight.push(vk::Fence::null()));

        let sync_objects = SyncObjects {
            image_available_semaphores,
            render_finished_semaphores,
            in_flight_fences,
            images_in_flight,
        };

        Ok(sync_objects)
    }

    fn update_command_buffer(
        device: &Device,
        command_context: &mut CommandContext,
        swapchain_context: &SwapchainContext,
        pipeline_context: &PipelineContext,
        render_pass_context: &RenderPassContext,
        framebuffers: &Vec<Framebuffer>,
        image_index: usize,
    ) -> Result<()> {
        let command_pool = command_context.command_pool;
        let command_buffer = command_context.command_buffer_per_image[image_index];
        unsafe { device.free_command_buffers(command_context.command_pool, &[command_buffer]) };

        let allocate_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&allocate_info) }?[0];
        command_context.command_buffer_per_image[image_index] = command_buffer;

        let inheritance_info = vk::CommandBufferInheritanceInfo::default();

        let info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT)
            .inheritance_info(&inheritance_info);

        unsafe { device.begin_command_buffer(command_buffer, &info) }?;
        let render_area = vk::Rect2D::default()
            .offset(vk::Offset2D::default())
            .extent(swapchain_context.swapchain_extent);

        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0],
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass_context.render_pass)
            .framebuffer(framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        unsafe { device.cmd_begin_render_pass(command_buffer, &info, vk::SubpassContents::INLINE) };
        unsafe {
            device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline_context.graphics_pipeline,
            )
        };

        // unsafe { device.cmd_bind_vertex_buffers(command_buffer, 0, &[], &[0]) };

        unsafe { device.cmd_draw(command_buffer, 3, 1, 0, 0) };
        unsafe { device.cmd_end_render_pass(command_buffer) };
        unsafe { device.end_command_buffer(command_buffer) }?;

        Ok(())
    }

    fn create_vertex_buffer() -> Result<()> {
        Ok(())
    }
}

impl Drop for Vulkan {
    fn drop(&mut self) {
        debug!("Drop Vulkan");
        unsafe {
            self.device_context
                .device
                .destroy_command_pool(self.command_context.command_pool, None)
        };

        unsafe {
            self.device_context
                .device
                .destroy_pipeline(self.pipeline_context.graphics_pipeline, None)
        };
        unsafe {
            self.device_context
                .device
                .destroy_pipeline_layout(self.pipeline_context.pipeline_layout, None)
        };

        unsafe {
            self.device_context
                .device
                .destroy_descriptor_set_layout(self.pipeline_context.descriptor_set_layout, None)
        };

        unsafe {
            self.device_context
                .device
                .destroy_render_pass(self.render_pass_context.render_pass, None)
        };

        self.swapchain_context
            .swapchain_image_views
            .iter()
            .for_each(|i| unsafe { self.device_context.device.destroy_image_view(*i, None) });

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
