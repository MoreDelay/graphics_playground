#![allow(
    dead_code,
    unused_variables,
    clippy::too_many_arguments,
    clippy::unnecessary_wraps
)]

use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::CStr;
use std::fs;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{BufReader, Read};
use std::mem::size_of;
use std::os::raw::c_void;
use std::path::PathBuf;
use std::ptr::copy_nonoverlapping as memcpy;
use std::time::Instant;

use vulkanalia::bytecode::Bytecode;
use vulkanalia::loader::{LibloadingLoader, LIBRARY};
use vulkanalia::prelude::v1_0::*;
use vulkanalia::vk::{ExtDebugUtilsExtension, KhrSurfaceExtension, KhrSwapchainExtension};
use vulkanalia::{window as vk_window, Version};

use anyhow::{anyhow, Result};
use cgmath::{point3, vec2, vec3, Deg};
use log::*;
use thiserror::Error;
use winit::window::Window;

const PORTABILITY_MACOS_VERSION: Version = Version::new(1, 3, 216);

const VALIDATION_ENABLED: bool = cfg!(debug_assertions);
const VALIDATION_LAYER: vk::ExtensionName =
    vk::ExtensionName::from_bytes(b"VK_LAYER_KHRONOS_validation");

const DEVICE_EXTENSIONS: &[vk::ExtensionName] = &[vk::KHR_SWAPCHAIN_EXTENSION.name];

const MAX_FRAMES_IN_FLIGHT: usize = 2;

// cgmath provides us with data types that directly map to shader data types.
type Vec2 = cgmath::Vector2<f32>;
type Vec3 = cgmath::Vector3<f32>;
type Mat4 = cgmath::Matrix4<f32>;

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct Vertex {
    pos: Vec3,
    color: Vec3,
    tex_coord: Vec2,
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Self) -> bool {
        self.pos == other.pos && self.color == other.color && self.tex_coord == other.tex_coord
    }
}

impl Eq for Vertex {}

impl Hash for Vertex {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.pos[0].to_bits().hash(state);
        self.pos[1].to_bits().hash(state);
        self.pos[2].to_bits().hash(state);
        self.color[0].to_bits().hash(state);
        self.color[1].to_bits().hash(state);
        self.color[2].to_bits().hash(state);
        self.tex_coord[0].to_bits().hash(state);
        self.tex_coord[1].to_bits().hash(state);
    }
}

impl Vertex {
    const fn new(pos: Vec3, color: Vec3, tex_coord: Vec2) -> Self {
        Self {
            pos,
            color,
            tex_coord,
        }
    }

    fn binding_description() -> vk::VertexInputBindingDescription {
        // describes the rate to load data from memory, i.e. number of bytes per entry and whether
        // to move to next data after each vertex or after each instance.
        vk::VertexInputBindingDescription::builder()
            .binding(0) // index in array of bindings
            .stride(size_of::<Vertex>() as u32) // entry size
            .input_rate(vk::VertexInputRate::VERTEX) // VERTEX or INSTANCE
            .build()
    }

    fn attribute_description() -> [vk::VertexInputAttributeDescription; 3] {
        // Description on how to extract a vertex attribute from a chunk of vertex data.
        // The data format is described with the same labels as color formats, by convention:
        // f32 -> R32_SFLOAT
        // Vector2<f32> -> R32G32_SFLOAT
        // Vector3<f32> -> R32G32B32_SFLOAT
        // Vector4<f32> -> R32G32B32A32_SFLOAT
        // Vector2<i32> -> R32G32_SINT
        // Vector4<u32> -> R32G32B32A32_UINT
        // Providing more channels than needed discards them silently. Providing less channels than
        // needed will give you the default values in the shader: 0 for color, 1 for alpha channel.
        let pos = vk::VertexInputAttributeDescription::builder()
            .binding(0) // index in array of bindings
            .location(0) // location index of shader
            .format(vk::Format::R32G32B32_SFLOAT) // data format, using color format specs
            .offset(0) // bytes since start of entry
            .build();

        let color = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(1)
            .format(vk::Format::R32G32B32_SFLOAT)
            .offset(size_of::<Vec3>() as u32)
            .build();

        let tex_coord = vk::VertexInputAttributeDescription::builder()
            .binding(0)
            .location(2)
            .format(vk::Format::R32G32_SFLOAT)
            .offset((size_of::<Vec3>() + size_of::<Vec3>()) as u32)
            .build();

        [pos, color, tex_coord]
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
struct UniformBufferObject {
    // We need to take care for alignment here. Rust has no good support for making sure the
    // members have correct alignment. In that case, you need to add some padding members.
    foo: Vec2, // 8 bytes, but we need to align to a multiple of 16 for struct members
    _padding: [u8; 8], // add 8 bytes of padding
    // Upload model matrix using push constants.
    view: Mat4,
    proj: Mat4,
}

// Our Vulkan App
#[derive(Clone, Debug)]
pub struct Vulkan {
    entry: Entry,
    instance: Instance,
    data: VulkanData,
    device: Device,
    frame: usize,

    resized: bool,
    start: Instant,

    models: usize,
}

impl Vulkan {
    // Create out Vulkan app.
    pub fn new(window: &Window) -> Result<Self> {
        unsafe {
            let loader = LibloadingLoader::new(LIBRARY)?;
            let entry = Entry::new(loader).map_err(|b| anyhow!("{}", b))?;
            let mut data = VulkanData::default();
            let instance = create_instance(window, &entry, &mut data)?;

            // Window creation is not necessary, e.g. for off-screen computation to a file. As we want
            // to render to a screen, we need to create a surface as render target.

            // // This is what actually happens to create a platform specific surface. The specific
            // // command looks different on all platforms.
            // let display_ptr = if let RawDisplayHandle::Wayland(display) = window.display_handle()?.as_raw() {
            //     display.display.as_ptr()
            // } else {
            //     panic!("Not a wayland handle!");
            // };
            //
            // let surface_ptr = if let RawWindowHandle::Wayland(window) = window.window_handle()?.as_raw() {
            //     window.surface.as_ptr()
            // } else {
            //     panic!("Not a wayland handle!");
            // };
            //
            // let info = vk::WaylandSurfaceCreateInfoKHR::builder()
            //     .display(display_ptr)
            //     .surface(surface_ptr);
            //
            // data.surface = instance.create_wayland_surface_khr(&info, none)?;

            // There is also this helper function from vulkanalia that looks the same for all platforms
            // but performs the correct platform specific commands.
            data.surface = vk_window::create_surface(&instance, &window, &window)?;

            pick_physical_device(&instance, &mut data)?;
            let device = create_logical_device(&entry, &instance, &mut data)?;

            create_swapchain(window, &instance, &device, &mut data)?;
            create_swapchain_image_views(&device, &mut data)?;

            create_render_pass(&instance, &device, &mut data)?;
            create_descriptor_set_layout(&device, &mut data)?;
            create_pipeline(&device, &mut data)?;

            create_command_pools(&instance, &device, &mut data)?;

            create_color_objects(&instance, &device, &mut data)?;
            create_depth_object(&instance, &device, &mut data)?;
            create_framebuffers(&device, &mut data)?;

            create_texture_image(&instance, &device, &mut data)?;
            create_texture_image_view(&device, &mut data)?;
            create_texture_sampler(&device, &mut data)?;

            load_model(&mut data)?;
            create_vertex_buffer(&instance, &device, &mut data)?;
            create_index_buffer(&instance, &device, &mut data)?;

            create_uniform_buffers(&instance, &device, &mut data)?;
            create_descriptor_pool(&device, &mut data)?;
            create_descriptor_sets(&device, &mut data)?;

            create_command_buffers(&device, &mut data)?;

            create_sync_objects(&device, &mut data)?;

            Ok(Self {
                entry,
                instance,
                data,
                device,
                frame: 0,
                resized: false,
                start: Instant::now(),
                models: 1,
            })
        }
    }

    // Renders a frame for our Vulkan app.
    pub fn render(&mut self, window: &Window) -> Result<()> {
        unsafe {
            // Wait for the previous frame of this fence to finish.
            self.device.wait_for_fences(
                &[self.data.in_flight_fences[self.frame]],
                true,
                u64::MAX,
            )?;

            // First acquire an image from the swapchain. As the swapchain is an extension feature, we
            // use a function with the suffix _khr. Using synchronization objects to signal when the
            // operation is completed.
            // Vulkan will let us know here if the swapchain is no longer adequat for the surface.
            let result = self.device.acquire_next_image_khr(
                self.data.swapchain,
                u64::MAX, // timeout in nanoseconds when image becomes available (disabled here)
                self.data.image_available_semaphores[self.frame], // semaphore for sync
                vk::Fence::null(), // fence for sync
            );

            let image_index = match result {
                // Ok path also includes suboptimal swapchain (SuccessCode).
                Ok((image_index, _)) => image_index as usize,
                Err(vk::ErrorCode::OUT_OF_DATE_KHR) => return self.recreate_swapchain(window),
                Err(e) => return Err(anyhow!(e)),
            };

            // If the swapchain returns images out of order (or MAX_FRAMES_IN_FLIGHT is larger than the
            // number of images in the swapchain), we might render into the wrong image that is
            // actually already in flight. Keep track of this by referencing the correct fence.
            if !self.data.images_in_flight[image_index as usize].is_null() {
                self.device.wait_for_fences(
                    &[self.data.images_in_flight[image_index as usize]],
                    true,
                    u64::MAX,
                )?;
            }

            self.data.images_in_flight[image_index as usize] =
                self.data.in_flight_fences[self.frame];

            // Update our uniform / global matrices and push constants. We need to be careful not to
            // overwrite a resource that is still used to draw another image. This is why we do changes
            // to these resources after the above fence.
            self.update_command_buffer(image_index)?;
            self.update_uniform_buffer(image_index)?;

            let wait_semaphores = &[self.data.image_available_semaphores[self.frame]];
            let wait_stages = &[vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let command_buffers = &[self.data.command_buffers[image_index as usize]];
            let signal_semaphores = &[self.data.render_finished_semaphores[self.frame]];
            let submit_info = vk::SubmitInfo::builder()
                // For synchronization, specify a semaphore and the stage where to wait for a signal.
                .wait_semaphores(wait_semaphores) // wait for these before output
                .wait_dst_stage_mask(wait_stages) // wait at these stages
                .command_buffers(command_buffers) // commands to submit for execution
                .signal_semaphores(signal_semaphores); // signal to these that command buffer is done

            // We need to manually reset fences to the unsignaled state. Do this right before we use
            // it.
            self.device
                .reset_fences(&[self.data.in_flight_fences[self.frame]])?;

            // Submit command buffer to the graphics queue. Here we can signal a fence that command
            // buffer is finished if we were using them.
            // Using fence to synchronize CPU with GPU.
            self.device.queue_submit(
                self.data.graphics_queue,
                &[submit_info],
                self.data.in_flight_fences[self.frame],
            )?;

            // While we only have a single subpass within the render queue, all commands before and all
            // commands after the subpass are each implicitly also a subpass. The starting subpass
            // assumes we can begin with image transition right away while we actually do not yet have
            // acquired the image. One way to solve this issue is to use the TOP_OF_PIPE flag for
            // wait_stages, or solve the issue with subpass dependencies. This is done in the
            // create_render_pass function.

            let swapchains = &[self.data.swapchain];
            let image_indices = &[image_index as u32];
            let present_info = vk::PresentInfoKHR::builder()
                .wait_semaphores(signal_semaphores) // wait on these before presenting
                .swapchains(swapchains) // target swapchain to present image to
                .image_indices(image_indices); // image to present to (almost always just one)
                                               // .results(None); // If multiple swapchains involved,
                                               // can check each individually here

            // Submit present request to swapchain. Errors here do not necessarily mean we need to
            // terminate the program.
            let result = self
                .device
                .queue_present_khr(self.data.present_queue, &present_info);

            // Also check for suboptimal to handle that at least here.
            let changed = result == Ok(vk::SuccessCode::SUBOPTIMAL_KHR)
                || result == Err(vk::ErrorCode::OUT_OF_DATE_KHR);

            // Also check for explicit resize. Do it after present so that semaphore state is consistent.
            if changed || self.resized {
                self.resized = false;
                self.recreate_swapchain(window)?;
            } else if let Err(e) = result {
                return Err(anyhow!(e));
            }

            // When we do not wait for the GPU to finish presenting and the CPU is faster in submitting
            // more work, we will slowly fill up memory with queued up commands. A simple solution is
            // to wait for the present queue to be idle again, but this is not using resources
            // efficiently. Instead, we can have multiple frames in flight and keep track of the
            // correct semaphore pairing with a frame field.
            // self.device.queue_wait_idle(self.data.present_queue)?;

            self.frame = (self.frame + 1) % MAX_FRAMES_IN_FLIGHT;
            Ok(())
        }
    }

    // Destroys our Vulkan app.
    // Destruction should happen in opposite order than creation.
    unsafe fn destroy(&mut self) {
        self.device.device_wait_idle().unwrap();

        self.destroy_swapchain();

        self.data
            .command_pools
            .iter()
            .for_each(|p| self.device.destroy_command_pool(*p, None));

        self.device.destroy_sampler(self.data.texture_sampler, None);
        self.device
            .destroy_image_view(self.data.texture_image_view, None);
        self.device.destroy_image(self.data.texture_image, None);
        self.device
            .free_memory(self.data.texture_image_memory, None);

        self.device
            .destroy_descriptor_set_layout(self.data.descriptor_set_layout, None);

        self.device.destroy_buffer(self.data.index_buffer, None);
        self.device.free_memory(self.data.index_buffer_memory, None);
        self.device.destroy_buffer(self.data.vertex_buffer, None);
        self.device
            .free_memory(self.data.vertex_buffer_memory, None);

        self.data
            .in_flight_fences
            .iter()
            .for_each(|f| self.device.destroy_fence(*f, None));

        self.data
            .render_finished_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));
        self.data
            .image_available_semaphores
            .iter()
            .for_each(|s| self.device.destroy_semaphore(*s, None));

        self.device
            .destroy_command_pool(self.data.transfer_command_pool, None);
        self.device
            .destroy_command_pool(self.data.command_pool, None);
        self.instance.destroy_surface_khr(self.data.surface, None);

        // Destroy logical device handler
        self.device.destroy_device(None);

        // Destroy callback handler
        if VALIDATION_ENABLED {
            self.instance
                .destroy_debug_utils_messenger_ext(self.data.messenger, None);
        }

        // The Vulkan instance should be destroyed last in the application.
        self.instance.destroy_instance(None);
    }

    unsafe fn destroy_swapchain(&mut self) {
        self.device
            .destroy_descriptor_pool(self.data.descriptor_pool, None);

        self.device
            .destroy_image_view(self.data.depth_image_view, None);
        self.device.free_memory(self.data.depth_image_memory, None);
        self.device.destroy_image(self.data.depth_image, None);

        self.device
            .destroy_image_view(self.data.color_image_view, None);
        self.device.free_memory(self.data.color_image_memory, None);
        self.device.destroy_image(self.data.color_image, None);

        self.data
            .uniform_buffers
            .iter()
            .for_each(|b| self.device.destroy_buffer(*b, None));
        self.data
            .uniform_buffers_memory
            .iter()
            .for_each(|m| self.device.free_memory(*m, None));

        // Command buffers can be reused. This is no longer needed as we reset the whole command
        // pool instead.
        // self.device
        //     .free_command_buffers(self.data.command_pool, &self.data.command_buffers);

        self.data
            .framebuffers
            .iter()
            .for_each(|f| self.device.destroy_framebuffer(*f, None));
        self.device.destroy_pipeline(self.data.pipeline, None);
        self.device
            .destroy_pipeline_layout(self.data.pipeline_layout, None);
        self.device.destroy_render_pass(self.data.render_pass, None);

        // Unlike images themselves, image views were created by us, so we need to destroy them.
        self.data
            .swapchain_image_views
            .iter()
            .for_each(|v| self.device.destroy_image_view(*v, None));

        self.device.destroy_swapchain_khr(self.data.swapchain, None);
    }

    // When the current swapchain no longer first fits the window surface, e.g. on resize, we
    // need to recreate the swapchain. Destroy the swapchain completely means we have to stop
    // rendering. Instead, we can also pass in the old swapchain into the info struct that we use
    // to create the new swapchain. After that, we need to destroy the old swapchain as soon as we
    // are done with all in-flight rendering tasks.
    unsafe fn recreate_swapchain(&mut self, window: &Window) -> Result<()> {
        // Do not touch resources while in use.
        self.device.device_wait_idle()?;
        self.destroy_swapchain();

        create_swapchain(window, &self.instance, &self.device, &mut self.data)?;
        create_swapchain_image_views(&self.device, &mut self.data)?;
        // Image format probably will not change on window resize, but still handle that case.
        create_render_pass(&self.instance, &self.device, &mut self.data)?;
        // We can avoid pipeline recreation by using dynamic state for viewport and scissor size.
        create_pipeline(&self.device, &mut self.data)?;
        // Recreate MSAA objects.
        create_color_objects(&self.instance, &self.device, &mut self.data)?;
        // Recreate depth objects.
        create_depth_object(&self.instance, &self.device, &mut self.data)?;
        // Framebuffers depend directly on image views.
        create_framebuffers(&self.device, &mut self.data)?;
        // Same as uniform buffers.
        create_uniform_buffers(&self.instance, &self.device, &mut self.data)?;
        // And descriptor pool.
        create_descriptor_pool(&self.device, &mut self.data)?;
        create_descriptor_sets(&self.device, &mut self.data)?;
        // Commands will need to render to a different area now.
        create_command_buffers(&self.device, &mut self.data)?;
        // There is a chance that we have a different number of images with the new swapchain.
        self.data
            .images_in_flight
            .resize(self.data.swapchain_images.len(), vk::Fence::null());

        Ok(())
    }

    // Make the model spin around based on time.
    unsafe fn update_uniform_buffer(&self, image_index: usize) -> Result<()> {
        // Set camera position using right handed coordinate system (as used by Vulkan).
        // Looking at the model from an 45 Degree angle.
        let view = Mat4::look_at_rh(
            point3(6.0, 0.0, 2.0), // eye position
            point3(0.0, 0.0, 0.0), // center position
            vec3(0.0, 0.0, 1.0),   // up axis
        );

        let aspect_ratio =
            self.data.swapchain_extent.width as f32 / self.data.swapchain_extent.height as f32;
        let mut proj = cgmath::perspective(
            Deg(45.0),    // field of view
            aspect_ratio, // aspect ratio
            0.1,          // near plane
            10.0,         // far plane
        );
        // cgmath was designed for OpenGL, which has the Y coordinate flipped. Furthermore, Vulkan
        // uses depth values in the range of [0, 1] while OpenGL uses [-1, 1], so we fix that.
        #[rustfmt::skip]
        let correction = Mat4::new( // column-major order initialization!
            1.0,  0.0, 0.0, 0.0,
            0.0, -1.0, 0.0, 0.0, // flip y axis
            0.0,  0.0, 0.5, 0.0, // transform depth value
            0.0,  0.0, 0.5, 1.0,
        );
        proj = correction * proj;

        // let ubo = UniformBufferObject { model, view, proj };
        let foo = vec2(0.0, 0.0);
        let _padding = [0; 8];
        let ubo = UniformBufferObject {
            foo,
            _padding,
            view,
            proj,
        };

        // Map our uniform buffers to the device without staging, because we change it so often.
        // This is wasteful because we only have a small change but transfer a lot of data. More
        // efficiently we would use push constants that hold only little changing data.
        let memory = self.device.map_memory(
            self.data.uniform_buffers_memory[image_index],
            0,
            size_of::<UniformBufferObject>() as u64,
            vk::MemoryMapFlags::empty(),
        )?;

        memcpy(&ubo, memory.cast(), 1);

        self.device
            .unmap_memory(self.data.uniform_buffers_memory[image_index]);

        Ok(())
    }

    unsafe fn update_command_buffer(&mut self, image_index: usize) -> Result<()> {
        // Option: Reset whole command pool and with it all command buffers.
        let command_pool = self.data.command_pools[image_index];
        self.device
            .reset_command_pool(command_pool, vk::CommandPoolResetFlags::empty())?;

        let command_buffer = self.data.command_buffers[image_index];

        // Option: Create a completely new command buffer to update values.
        // let previous = self.data.command_buffers[image_index];
        // self.device
        //     .free_command_buffers(self.data.command_pool, &[previous]);
        //
        // let allocate_info = vk::CommandBufferAllocateInfo::builder()
        //     .command_pool(self.data.command_pool)
        //     .level(vk::CommandBufferLevel::PRIMARY)
        //     .command_buffer_count(1);
        //
        // let command_buffer = self.device.allocate_command_buffers(&allocate_info)?[0];
        // self.data.command_buffers[image_index] = command_buffer;

        // // Option: Reset command buffer to record new commands.
        // let command_buffer = self.data.command_buffers[image_index];
        // self.device
        //     .reset_command_buffer(command_buffer, vk::CommandBufferResetFlags::empty())?;

        // This is only relevant for secondary command buffers that specifies what state to inherit
        // from the calling primary command buffer
        let inheritance = vk::CommandBufferInheritanceInfo::builder();

        // Flags can be the following:
        // ONE_TIME_SUBMIT: rerecord command buffer after one execution
        // RENDER_PASS_CONTINUE: secondary command buffer only used within single render pass
        // SIMULTANEOUS_USE: can resubmit command buffer while already pending execution
        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT) // needs update for push constants
            .inheritance_info(&inheritance);

        // If a command buffer was already recorded, it gets implicitly reset by
        // begin_command_buffer. You can not append commands to a buffer at a later time.
        self.device.begin_command_buffer(command_buffer, &info)?;

        // The render area defines the area that is affected by the command buffer. Outside this area
        // pixels will have undefined values, so match it with the attachment size.
        let render_area = vk::Rect2D::builder()
            .offset(vk::Offset2D::default())
            .extent(self.data.swapchain_extent);

        // This is the value that framebuffers are set to before render pass (because we specified
        // to clear attachments on the render pass).
        let color_clear_value = vk::ClearValue {
            color: vk::ClearColorValue {
                float32: [0.0, 0.0, 0.0, 1.0], // Black with 100% opacity
            },
        };

        let depth_clear_value = vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue {
                depth: 1.0,
                stencil: 0,
            },
        };

        // Drawing starts by calling cmd_begin_render_pass. We need to know the render pass to
        // start and their attachments, and the framebuffer that specifies each image as color
        // attachment.
        let clear_values = &[color_clear_value, depth_clear_value];
        let info = vk::RenderPassBeginInfo::builder()
            .render_pass(self.data.render_pass)
            .framebuffer(self.data.framebuffers[image_index])
            .render_area(render_area)
            .clear_values(clear_values);

        // Render pass can now begin. All functions that record a command use the prefix cmd_ and
        // return (), error handling is done after all recording is done. Here we issue a command
        // to start the render pass. The first command will always bind the associated framebuffer,
        // so we need a different command buffer for each image.
        // We can provide drawing commands in two ways:
        // INLINE: render pass commands are embedded in primary command buffer, no secondary buffer
        // SECONDARY_COMMAND_BUFFERS: render pass commands are executed from secondary buffers
        // You can not mix both, either all commands are in the primary or only secondary buffers.
        self.device.cmd_begin_render_pass(
            command_buffer,
            &info,
            vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
        );

        // All command get defined in a secondary command buffer.
        let secondary_command_buffers = (0..self.models)
            .map(|i| self.update_secondary_command_buffer(image_index, i))
            .collect::<Result<Vec<_>, _>>()?;
        self.device
            .cmd_execute_commands(command_buffer, &secondary_command_buffers[..]);

        // Finish up
        self.device.cmd_end_render_pass(command_buffer);

        // Complete recording to the command buffer. Here we do error checking.
        self.device.end_command_buffer(command_buffer)?;

        Ok(())
    }

    unsafe fn update_secondary_command_buffer(
        &mut self,
        image_index: usize,
        model_index: usize,
    ) -> Result<vk::CommandBuffer> {
        let command_buffers = &mut self.data.secondary_command_buffers[image_index];

        // Allocate on-demand as many command buffers as you need or reuse the existing ones.
        while model_index >= command_buffers.len() {
            let allocate_info = vk::CommandBufferAllocateInfo::builder()
                .command_pool(self.data.command_pools[image_index])
                .level(vk::CommandBufferLevel::SECONDARY)
                .command_buffer_count(1);

            let command_buffer = self.device.allocate_command_buffers(&allocate_info)?[0];
            command_buffers.push(command_buffer);
        }

        let command_buffer = command_buffers[model_index];

        // Secondary command buffers need to know from where they will inherit state.
        let inheritance_info = vk::CommandBufferInheritanceInfo::builder()
            .render_pass(self.data.render_pass)
            .subpass(0)
            .framebuffer(self.data.framebuffers[image_index]); // only for driver optimization

        let info = vk::CommandBufferBeginInfo::builder()
            .flags(vk::CommandBufferUsageFlags::RENDER_PASS_CONTINUE) // only used in render pass
            .inheritance_info(&inheritance_info);

        self.device.begin_command_buffer(command_buffer, &info)?;

        let move_y = (model_index % 2) as f32 * 2.5 - 1.25;
        let move_z = (model_index / 2) as f32 * -2.0 + 1.0;

        let time = self.start.elapsed().as_secs_f32();

        // Create rotation matrix from axis and angle.
        let model = Mat4::from_translation(vec3(0.0, move_y, move_z))
            * Mat4::from_axis_angle(vec3(0.0, 0.0, 1.0), Deg(90.0) * time);
        let model_bytes =
            std::slice::from_raw_parts(&model as *const Mat4 as *const u8, size_of::<Mat4>());

        let opacity = 0.25 * (model_index + 1) as f32;

        // Bind the pipeline as graphics pipeline.
        self.device.cmd_bind_pipeline(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.data.pipeline,
        );

        // Bind our vertex buffer.
        self.device.cmd_bind_vertex_buffers(
            command_buffer,
            0,                          // index of vertex input binding
            &[self.data.vertex_buffer], // vertex buffer to bind
            &[0],                       // offset into buffer to start reading from
        );

        // Bind out index buffer. At each point there can only be a single index buffer, so for
        // differing vertex attributes you will have to duplicate all vertex self.data.
        self.device.cmd_bind_index_buffer(
            command_buffer,
            self.data.index_buffer,
            0,
            vk::IndexType::UINT32,
        );

        // Bind descriptor sets for uniform buffer.
        self.device.cmd_bind_descriptor_sets(
            command_buffer,
            vk::PipelineBindPoint::GRAPHICS, // descriptors can also be used in compute pipelines
            self.data.pipeline_layout,
            0,                                         // first descriptor set in an array
            &[self.data.descriptor_sets[image_index]], // descriptor sets to bind for array
            &[],                                       // offsets for dynamic descriptors
        );

        // Push constant values to shader.
        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::VERTEX,
            0,
            model_bytes,
        );

        self.device.cmd_push_constants(
            command_buffer,
            self.data.pipeline_layout,
            vk::ShaderStageFlags::FRAGMENT,
            size_of::<Mat4>() as u32,
            &opacity.to_ne_bytes()[..],
        );

        // The actual draw command using an index buffer.
        self.device.cmd_draw_indexed(
            command_buffer,
            self.data.indices.len() as u32, // index count
            1,                              // instance count
            0,                              // first vertex
            0,                              // vertex offset
            0,                              // first instance
        );

        self.device.end_command_buffer(command_buffer)?;

        Ok(command_buffer)
    }

    pub(crate) fn add_model(&mut self, arg: usize) {
        self.models = usize::min(self.models + arg, 4);
    }

    pub(crate) fn sub_model(&mut self, arg: usize) {
        self.models = usize::max(self.models - arg, 1);
    }
}

// The Vulkan handles and associated properties used by our Vulkan app.
#[derive(Clone, Debug, Default)]
struct VulkanData {
    // callback function as a handler
    messenger: vk::DebugUtilsMessengerEXT,
    // gpu handle
    physical_device: vk::PhysicalDevice,
    // handle for window surface
    surface: vk::SurfaceKHR,
    // logical device queue handles
    graphics_queue: vk::Queue,
    present_queue: vk::Queue,
    transfer_queue: vk::Queue,
    // our swapchain
    swapchain: vk::SwapchainKHR,
    swapchain_images: Vec<vk::Image>,
    swapchain_format: vk::Format,
    swapchain_extent: vk::Extent2D,
    swapchain_image_views: Vec<vk::ImageView>,
    // pipeline
    render_pass: vk::RenderPass,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    // actual framebuffers
    framebuffers: Vec<vk::Framebuffer>,
    // commands
    command_pool: vk::CommandPool,
    transfer_command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
    // synchronization
    image_available_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphores: Vec<vk::Semaphore>,
    in_flight_fences: Vec<vk::Fence>,
    images_in_flight: Vec<vk::Fence>,
    // vertex data
    vertex_buffer: vk::Buffer,
    vertex_buffer_memory: vk::DeviceMemory,
    index_buffer: vk::Buffer,
    index_buffer_memory: vk::DeviceMemory,
    // uniform data
    descriptor_set_layout: vk::DescriptorSetLayout,
    uniform_buffers: Vec<vk::Buffer>,
    uniform_buffers_memory: Vec<vk::DeviceMemory>,
    descriptor_pool: vk::DescriptorPool,
    descriptor_sets: Vec<vk::DescriptorSet>,
    // texture
    texture_image: vk::Image,
    texture_image_memory: vk::DeviceMemory,
    texture_image_view: vk::ImageView,
    texture_sampler: vk::Sampler,
    // depth
    depth_image: vk::Image,
    depth_image_memory: vk::DeviceMemory,
    depth_image_view: vk::ImageView,
    // mesh
    vertices: Vec<Vertex>,
    indices: Vec<u32>,
    // mipmap
    mip_levels: u32,
    // Multi-Sample Anti-Aliasing (MSAA)
    msaa_samples: vk::SampleCountFlags,
    color_image: vk::Image,
    color_image_memory: vk::DeviceMemory,
    color_image_view: vk::ImageView,
    // multiple instance
    command_pools: Vec<vk::CommandPool>,
    secondary_command_buffers: Vec<Vec<vk::CommandBuffer>>,
}

#[derive(Copy, Clone, Debug)]
struct QueueFamilyIndices {
    // Index of a queue with graphics support.
    graphics: u32,
    // Index of a queue with presentation support. Can be different to the graphics queue.
    present: u32,
    // Index of a queue with transfer support. Not really needed as graphics can take on that role.
    transfer: u32,
}

impl QueueFamilyIndices {
    unsafe fn get(
        instance: &Instance,
        data: &VulkanData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        let properties = instance.get_physical_device_queue_family_properties(physical_device);

        // Look for the first queue that support graphics commands.
        let graphics = properties
            .iter()
            .position(|p| p.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|i| i as u32);

        // Look for the first queue that supports present commands. May be different than the
        // graphics queue. Most of the time they should be the same queue, but you can explicitly
        // check for that as it is more performant when they are the same queue for present and
        // graphics.
        let present = properties
            .iter()
            .enumerate()
            .position(|(idx, prop)| {
                instance
                    .get_physical_device_surface_support_khr(
                        physical_device,
                        idx as u32,
                        data.surface,
                    )
                    .is_ok()
            })
            .map(|i| i as u32);

        // Transfer operations can be sent through the graphics queue. This is for learning
        // purposes to learn about concurrent queue operations.
        let graphics_id = if let Some(id) = graphics {
            id as u32
        } else {
            u32::MAX
        };

        let transfer = properties
            .iter()
            .enumerate()
            .position(|(idx, p)| {
                p.queue_flags.contains(vk::QueueFlags::TRANSFER) && (idx as u32 != graphics_id)
            })
            .map(|i| i as u32);

        if let (Some(graphics), Some(present), Some(transfer)) = (graphics, present, transfer) {
            Ok(Self {
                graphics,
                present,
                transfer,
            })
        } else {
            Err(anyhow!(SuitabilityError(
                "Missing required queue families."
            )))
        }
    }
}

#[derive(Clone, Debug)]
struct SwapchainSupport {
    capabilities: vk::SurfaceCapabilitiesKHR,
    formats: Vec<vk::SurfaceFormatKHR>,
    present_modes: Vec<vk::PresentModeKHR>,
}

impl SwapchainSupport {
    unsafe fn get(
        instance: &Instance,
        data: &VulkanData,
        physical_device: vk::PhysicalDevice,
    ) -> Result<Self> {
        Ok(Self {
            capabilities: instance
                .get_physical_device_surface_capabilities_khr(physical_device, data.surface)?,
            formats: instance
                .get_physical_device_surface_formats_khr(physical_device, data.surface)?,
            present_modes: instance
                .get_physical_device_surface_present_modes_khr(physical_device, data.surface)?,
        })
    }
}

// Create a Vulkan instance from an entry point.
// Type Instance is not to be confused with the raw vk::Instance. Vulkanalia wraps these instances
// to combine the raw Vulkan instance with the commands that created them.
unsafe fn create_instance(
    window: &Window,
    entry: &Entry,
    data: &mut VulkanData,
) -> Result<Instance> {
    // Add validation layers whenever we compile for debug.
    let available_layers = entry
        .enumerate_instance_layer_properties()?
        .iter()
        .map(|l| l.layer_name)
        .collect::<HashSet<_>>();

    if VALIDATION_ENABLED && !available_layers.contains(&VALIDATION_LAYER) {
        return Err(anyhow!("Validation layer requested but not supported."));
    }

    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        Vec::new()
    };

    // Optional information so that Vulkan can optimize driver usage for our application.
    // Engine refers to popular applications with specific optimizations, such as game engines.
    let application_info = vk::ApplicationInfo::builder()
        .application_name(b"Vulkan Tutorial\0")
        .application_version(vk::make_version(1, 0, 0))
        .engine_name(b"No Engine\0")
        .engine_version(vk::make_version(1, 0, 0))
        .api_version(vk::make_version(1, 0, 0));

    // Specify the global extensions and validation layers we want to use.
    // Also includes VK_KHR_surface extension to communicate with (abstract) window surfaces. While
    // the API is platform agnostic, the creation itself is not and also needs platform
    // specific extensions (also loaded here), e.g. Windows VK_KHR_win32_surface.
    let mut extensions = vk_window::get_required_instance_extensions(window)
        .iter()
        .map(|e| e.as_ptr())
        .collect::<Vec<_>>();

    // Setup callback for logging instead of printing to STDOUT.
    if VALIDATION_ENABLED {
        extensions.push(vk::EXT_DEBUG_UTILS_EXTENSION.name.as_ptr());
    }

    // MacOS is not fully conformant with the Vulkan spec,
    // so we need to enable the compatibility extensions in that case.
    let flags = if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        info!("Enabling extensions for macOS portability.");
        extensions.push(
            vk::KHR_GET_PHYSICAL_DEVICE_PROPERTIES2_EXTENSION
                .name
                .as_ptr(),
        );
        extensions.push(vk::KHR_PORTABILITY_ENUMERATION_EXTENSION.name.as_ptr());
        vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
    } else {
        vk::InstanceCreateFlags::empty()
    };

    let mut info = vk::InstanceCreateInfo::builder()
        .application_info(&application_info)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .flags(flags);

    // Make callback for all types of severities and all types of messages.
    // This sets all flag bits known to vulkanalia, where not all bits may be supported by
    // earlier versions. In that case, there might be some validation warnings about that. In
    // that case, try upgrading your VulkanSDK.

    let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::builder()
        .message_severity(vk::DebugUtilsMessageSeverityFlagsEXT::all())
        .message_type(
            vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
        )
        // .user_data(&user_data)  // optional mutable reference to your own user_data
        .user_callback(Some(debug_callback));

    if VALIDATION_ENABLED {
        // Chain structs with next pointers.
        // Make callback valid during Create... and Destroy... functions.
        info = info.push_next(&mut debug_info);
    }

    // Custom allcator will always be None in this tutorial.
    let instance = entry.create_instance(&info, None)?;

    if VALIDATION_ENABLED {
        // Register callback with Vulkan instance for all other debug messages.
        data.messenger = instance.create_debug_utils_messenger_ext(&debug_info, None)?;
    }

    Ok(instance)
}

// Callback for Vulkan validation layer, so we specify extern "system"
// To see these messages printed out, set the environment variable RUST_LOG to the desired level.
// returns: should the Vulkan call be arborted? Used to test validation layer itself.
extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    type_: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut c_void,
) -> vk::Bool32 {
    let data = unsafe { *data };
    let message = unsafe { CStr::from_ptr(data.message) }.to_string_lossy();

    if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
        error!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::WARNING {
        warn!("({:?}) {}", type_, message);
    } else if severity >= vk::DebugUtilsMessageSeverityFlagsEXT::INFO {
        debug!("({:?}) {}", type_, message);
    } else {
        trace!("({:?}) {}", type_, message);
    }

    vk::FALSE
}

// Each entry has a format and a color_space member. The format specifies color channels and types,
// such as vk::Format::B8G8R8A8_SRGB (8 bits per color channel in order BGRA). The color_space
// indicates if sRGB is supported.
fn get_swapchain_surface_format(formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    formats
        .iter()
        .find(|f| {
            f.format == vk::Format::B8G8R8A8_SRGB
                && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
        })
        .cloned()
        .unwrap_or_else(|| formats[0]) // We could also rank for "best" alternative format.
}

// Present mode sets the condition on how to show images to the screen. The possible options are
// IMMEDIATE, FIFO, FIFO_RELAXED, MAILBOX. FIFO is guaranteed to be available.
// IMMEDIATE: Images are sent directly to the screen, can result in tearing.
// FIFO: Swapchain is a queue where images from the front are displayed and later put into the
// back. Application waits when queue is full. Similar to vertical sync in games.
// FIFO_RELAXED: Same as FIFO, but if queue is empty on last vertical blank (refresh), the next
// ready image gets sent to the screen like in IMMEDIATE mode.
// MAILBOX: Same as FIFO, but when queue is full we start to replace other queued images. Also
// called "triple buffering", costs more energy.
fn get_swapchain_present_mode(present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    present_modes
        .iter()
        .find(|m| **m == vk::PresentModeKHR::MAILBOX)
        .cloned() // clone late is generally better
        .unwrap_or(vk::PresentModeKHR::FIFO)
}

// The swap extent is the resolution of the swapchain images that is close to the resolution of the
// target window. Vulkan tells us to match the window resolution to the extent resolution. Some
// window managers allow differences in these resolutions, which is shown by having the extent set
// to the maximum value. In that case, we choose the best resolution match within some bounds.
fn get_swapchain_extent(window: &Window, capabilities: vk::SurfaceCapabilitiesKHR) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D::builder()
            .width(window.inner_size().width.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ))
            .height(window.inner_size().height.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ))
            .build()
    }
}

// When handling a swapchain through differing queues (graphics and present not the same), we need
// to handle the specify how to handle commands arriving on different queues.
// EXCLUSIVE: A queue owns an image and must explicitly transfer over ownership. Best performance.
// CONCURRENT: Use image across queues without explicit ownership transfer.
unsafe fn create_swapchain(
    window: &Window,
    instance: &Instance,
    device: &Device,
    data: &mut VulkanData,
) -> Result<()> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let support = SwapchainSupport::get(instance, data, data.physical_device)?;

    let surface_format = get_swapchain_surface_format(&support.formats);
    let present_mode = get_swapchain_present_mode(&support.present_modes);
    let extent = get_swapchain_extent(window, support.capabilities);

    // Request at least one more image than minimum so that driver overhead is no bottleneck.
    let mut image_count = support.capabilities.min_image_count + 1;
    // If no upper limit exists, the maximum is set to 0.
    if support.capabilities.max_image_count != 0
        && image_count > support.capabilities.max_image_count
    {
        image_count = support.capabilities.max_image_count;
    }

    // When the queues are not the same, we will use CONCURRENT here so we do not have to deal with
    // ownership transfers at the moment.
    let mut queue_family_indices = vec![];
    let image_sharing_mode = if indices.graphics != indices.present {
        queue_family_indices.push(indices.graphics);
        queue_family_indices.push(indices.present);
        vk::SharingMode::CONCURRENT
    } else {
        vk::SharingMode::EXCLUSIVE
    };

    let info = vk::SwapchainCreateInfoKHR::builder()
        .surface(data.surface)
        .min_image_count(image_count)
        .image_format(surface_format.format)
        .image_color_space(surface_format.color_space)
        .image_extent(extent)
        .image_array_layers(1) // always 1 unless you develop stereoscopic 3D apps
        // We will render directly to the image.
        // When you render to separate images first for post-processing, use TRANSFER_DST
        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        .image_sharing_mode(image_sharing_mode)
        .queue_family_indices(&queue_family_indices)
        // Here you could specify some transformation like 90 degree rotation or flipping.
        .pre_transform(support.capabilities.current_transform)
        // Set alpha channel setting here for blending.
        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
        .present_mode(present_mode)
        // If set to true, we do not care about obscured pixels, e.g. behind another window.
        .clipped(true)
        // You need to pass your old swapchain in if it became invalid, e.g. due to window resizing.
        .old_swapchain(vk::SwapchainKHR::null());

    data.swapchain = device.create_swapchain_khr(&info, None)?;
    data.swapchain_images = device.get_swapchain_images_khr(data.swapchain)?;
    data.swapchain_format = surface_format.format;
    data.swapchain_extent = extent;

    Ok(())
}

unsafe fn check_physical_device_extensions(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    let extensions = instance
        .enumerate_device_extension_properties(physical_device, None)?
        .iter()
        .map(|e| e.extension_name)
        .collect::<HashSet<_>>();

    // Check that the GPU has a swap chain extension. This should be available based on the present
    // queue, but we will be explicit here.
    if DEVICE_EXTENSIONS.iter().all(|e| extensions.contains(e)) {
        Ok(())
    } else {
        Err(anyhow!(SuitabilityError(
            "Missing required device extensions."
        )))
    }
}

#[derive(Debug, Error)]
#[error("Missing {0}.")]
pub struct SuitabilityError(pub &'static str);

// Check for GPU capabilities.
unsafe fn check_physical_device(
    instance: &Instance,
    data: &VulkanData,
    physical_device: vk::PhysicalDevice,
) -> Result<()> {
    // Test for basic properties such as name, type and supported Vulkan versions.
    let properties = instance.get_physical_device_properties(physical_device);
    if properties.device_type != vk::PhysicalDeviceType::DISCRETE_GPU {
        return Err(anyhow!(SuitabilityError(
            "Only discrete GPUs are supported."
        )));
    }

    // Test for optional hardware features.
    let features = instance.get_physical_device_features(physical_device);
    if features.geometry_shader != vk::TRUE {
        return Err(anyhow!(SuitabilityError(
            "Missing geometry shader support."
        )));
    }
    if features.sampler_anisotropy != vk::TRUE {
        return Err(anyhow!(SuitabilityError("No sampler anisotropy.")));
    }
    // You can make a ranking of your suitable devices and pick the best, or let the user decide.

    // Check we have a device with graphics support.
    QueueFamilyIndices::get(instance, data, physical_device)?;

    // Check for additional necessary extensions of the device
    check_physical_device_extensions(instance, physical_device)?;

    // Checking for swapchain support alone is not enough, we need to query its properties.
    let support = SwapchainSupport::get(instance, data, physical_device)?;
    if support.formats.is_empty() || support.present_modes.is_empty() {
        return Err(anyhow!(SuitabilityError("Insufficient swapchain support.")));
    }
    Ok(())
}

// Pick GPU when it satisfies our requirements.
unsafe fn pick_physical_device(instance: &Instance, data: &mut VulkanData) -> Result<()> {
    for physical_device in instance.enumerate_physical_devices()? {
        let properties = instance.get_physical_device_properties(physical_device);

        if let Err(error) = check_physical_device(instance, data, physical_device) {
            warn!(
                "Skipping physical device (`{}`): {}",
                properties.device_name, error
            );
        } else {
            info!("Selected physical device (`{}`).", properties.device_name);
            data.physical_device = physical_device;
            data.msaa_samples = get_max_msaa_samples(instance, data);
            return Ok(());
        }
    }

    Err(anyhow!("Failed to find suitable physical device."))
}

unsafe fn create_logical_device(
    entry: &Entry,
    instance: &Instance,
    data: &mut VulkanData,
) -> Result<Device> {
    // You can only create a small number of queues per queue family and logical device.
    // The idea is to create commands buffers on different threads and submit them once in the main thread.
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;

    let mut unique_indices = HashSet::new();
    unique_indices.insert(indices.graphics);
    unique_indices.insert(indices.present);
    unique_indices.insert(indices.transfer);

    // The queue priority influences scheduling of command buffer execution.
    // We request a single queue (length of priority vector)
    let queue_priorities = &[1.0];
    // Create all Info structs at once
    let queue_infos = unique_indices
        .iter()
        .map(|i| {
            vk::DeviceQueueCreateInfo::builder()
                .queue_family_index(*i)
                .queue_priorities(queue_priorities)
        })
        .collect::<Vec<_>>();

    // Specify device specific layers and extensions.
    let layers = if VALIDATION_ENABLED {
        vec![VALIDATION_LAYER.as_ptr()]
    } else {
        vec![]
    };

    let mut extensions = DEVICE_EXTENSIONS
        .iter()
        .map(|n| n.as_ptr())
        .collect::<Vec<_>>();

    // Required by Vulkan SDK on macOS because it is not fully compliant.
    if cfg!(target_os = "macos") && entry.version()? >= PORTABILITY_MACOS_VERSION {
        extensions.push(vk::KHR_PORTABILITY_SUBSET_EXTENSION.name.as_ptr());
    }

    // Specify the features we want to use with the logical device, which we tested for beforehand.
    // We need to enable anisotropy for our texture sampler.
    let features = vk::PhysicalDeviceFeatures::builder()
        .sampler_anisotropy(true) // enable anisotropy, i.e. counteract undersampling
        .sample_rate_shading(true); // allow for sample shading, MMSA within textures

    let info = vk::DeviceCreateInfo::builder()
        .queue_create_infos(&queue_infos)
        .enabled_layer_names(&layers)
        .enabled_extension_names(&extensions)
        .enabled_features(&features);

    let device = instance.create_device(data.physical_device, &info, None)?;

    // Queues are created automatically, and because we only crated one queue, we use index 0.
    data.graphics_queue = device.get_device_queue(indices.graphics, 0);
    data.present_queue = device.get_device_queue(indices.present, 0);
    data.transfer_queue = device.get_device_queue(indices.transfer, 0);

    Ok(device)
}

// Create basic handles to images to make render targets easily accessible.
unsafe fn create_swapchain_image_views(device: &Device, data: &mut VulkanData) -> Result<()> {
    data.swapchain_image_views = data
        .swapchain_images
        .iter()
        .map(|i| {
            create_image_view(
                device,
                *i,
                data.swapchain_format,
                vk::ImageAspectFlags::COLOR,
                1, // rendered images have no mipmap
            )
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

unsafe fn create_shader_module(device: &Device, bytecode: &[u8]) -> Result<vk::ShaderModule> {
    // Vulkan expects to have its bytecode as u32, while Rust reads in bytecode as u8.
    // Vulkanalia provides a helper struct to make that translation easy.
    let bytecode = Bytecode::new(bytecode).unwrap();

    let info = vk::ShaderModuleCreateInfo::builder()
        .code_size(bytecode.code_size())
        .code(bytecode.code());

    Ok(device.create_shader_module(&info, None)?)
}

unsafe fn create_render_pass(
    instance: &Instance,
    device: &Device,
    data: &mut VulkanData,
) -> Result<()> {
    // In our simple pipeline, we just use a single color buffer.
    let color_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format) // should match with the swapchain images
        .samples(data.msaa_samples) // use multi sampling
        // Determines what to do with the data in attachment before rendering.
        // LOAD: preserve existing content
        // CLEAR: clear values to constants
        // DONT_CARE: existing values are undefined
        .load_op(vk::AttachmentLoadOp::CLEAR)
        // Determines what to do with the data in attachment after rendering.
        // STORE: store rendered content in memory
        // DONT_CARE: contents in framebuffer undefined after rendering
        .store_op(vk::AttachmentStoreOp::STORE)
        // Same options with the stencil buffer, which we do not use here.
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        // Image layout can be different depending on your use case on what happens next.
        // COLOR_ATTACHMENT_OPTIMAL: images used as color attachments
        // PRESENT_SRC_KHR: images presented with swapchain
        // TRANSFER_DST_OPTIMAL: images are destinations for memory copy
        .initial_layout(vk::ImageLayout::UNDEFINED) // layout before rendering starts
        .final_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL); // need to resolve samples

    let depth_stencil_attachment = vk::AttachmentDescription::builder()
        .format(get_depth_format(instance, data)?)
        .samples(data.msaa_samples)
        .load_op(vk::AttachmentLoadOp::CLEAR)
        .store_op(vk::AttachmentStoreOp::DONT_CARE) // we will not store the depth buffer
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_resolve_attachment = vk::AttachmentDescription::builder()
        .format(data.swapchain_format)
        .samples(vk::SampleCountFlags::_1)
        .load_op(vk::AttachmentLoadOp::DONT_CARE)
        .store_op(vk::AttachmentStoreOp::DONT_CARE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(vk::ImageLayout::UNDEFINED)
        .final_layout(vk::ImageLayout::PRESENT_SRC_KHR);

    // A render pass can consist of multiple subpasses run on the same framebuffers. Each subpass
    // uses references to the attachment they operate on.
    let color_attachment_ref = vk::AttachmentReference::builder()
        .attachment(0) // index in attachment description array
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    let depth_stencil_attachment_ref = vk::AttachmentReference::builder()
        .attachment(1)
        .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);

    let color_resolve_attachment_ref = vk::AttachmentReference::builder()
        .attachment(2)
        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);

    // As there may be compute subpass support in Vulkan at some point, we need to specify a
    // graphics pass. Index in the color_attachments array maps to GLSL location index.
    let color_attachments = &[color_attachment_ref];
    let resolve_attachments = &[color_resolve_attachment_ref];
    let subpass = vk::SubpassDescription::builder()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .color_attachments(color_attachments) // render targets
        .depth_stencil_attachment(&depth_stencil_attachment_ref) // only one depth buffer used
        .resolve_attachments(resolve_attachments);

    // other attachments:
    // .input_attachments(None) // input for shader
    // .resolve_attachments(None) // used for multisampling color attachments
    // .depth_stencil_attachment(None) // for depth and stencil data
    // .preserve_attachments(None) // not used by this subpass but needs to be preserved

    // Add subpass dependencies to deal with initial subpass not waiting for image availability.
    // To refert to the implicit subpass, use the special value SUBPASS_EXTERNAL. If it is the
    // source, we mean the initial subpass. If it is the destination, we mean the closing subpass.
    // The dependency is a DAG, the destination subpass must always have a higher index.
    let dependency = vk::SubpassDependency::builder()
        .src_subpass(vk::SUBPASS_EXTERNAL) // dependency subpass
        .dst_subpass(0) // dependent subpass
        // Wait until the src subpass is in output stage (swapchain finished reading).
        .src_stage_mask(
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        )
        .src_access_mask(vk::AccessFlags::empty())
        // Wait when the dst subpass is in output stage and needs to write.
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
    let info = vk::RenderPassCreateInfo::builder()
        .attachments(attachments)
        .subpasses(subpasses)
        .dependencies(dependencies);

    data.render_pass = device.create_render_pass(&info, None)?;

    Ok(())
}
fn read_shader(filename: &str) -> Vec<u8> {
    let out_dir = env::var_os("OUT_DIR").unwrap();
    let out_dir = PathBuf::from(out_dir);
    let path = out_dir.join(filename);
    let mut file = File::open(&path).expect("vertex shader not found");
    let metadata = fs::metadata(&path).expect("could not read file length");
    let mut shader = vec![0; metadata.len() as usize];
    file.read(&mut shader).expect("buffer overflow");
    shader
}

unsafe fn create_pipeline(device: &Device, data: &mut VulkanData) -> Result<()> {
    let vert = read_shader("vert.spv");
    let frag = read_shader("frag.spv");

    // Compiling and linking from SPIR-V to machine code happens not here but when we create the
    // actual pipeline. At that point, we do not need references to these anymore.
    let vert_shader_module = create_shader_module(device, &vert[..])?;
    let frag_shader_module = create_shader_module(device, &frag[..])?;

    let vert_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::VERTEX)
        .module(vert_shader_module)
        // .specialization_info(None) // used to set shader constants at shader compile time
        .name(b"main\0"); // the entrypoint function to invoke within the shader

    let frag_stage = vk::PipelineShaderStageCreateInfo::builder()
        .stage(vk::ShaderStageFlags::FRAGMENT)
        .module(frag_shader_module)
        .name(b"main\0");

    // Put the vertex data here.
    let binding_descriptions = &[Vertex::binding_description()];
    let attribute_descriptions = Vertex::attribute_description();
    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::builder()
        .vertex_binding_descriptions(binding_descriptions)
        .vertex_attribute_descriptions(&attribute_descriptions);

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::builder()
        .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
        .primitive_restart_enable(false);

    // The images of your swapchain will be our framebuffers, so stick to their dimensions.
    let viewport = vk::Viewport::builder()
        .x(0.0)
        .y(0.0)
        .width(data.swapchain_extent.width as f32)
        .height(data.swapchain_extent.height as f32)
        .min_depth(0.0)
        .max_depth(1.0);

    // In this tutorial we will always want to use the whole framebuffer to draw to.
    let scissor = vk::Rect2D::builder()
        .offset(vk::Offset2D { x: 0, y: 0 })
        .extent(data.swapchain_extent);

    // Some GPUs allow to use multiple viewports and rectangles, but this requires enabling that
    // feature during logical device creation. Still, we need to provide an array.
    let viewports = &[viewport];
    let scissors = &[scissor];
    let viewport_state = vk::PipelineViewportStateCreateInfo::builder()
        .viewports(viewports)
        .scissors(scissors);

    // Most of the rasterizer options also require enabling GPU features at device creation.
    let rasterization_state = vk::PipelineRasterizationStateCreateInfo::builder()
        .depth_clamp_enable(false) // clamp fragments to view frustrum instead of discarding
        .rasterizer_discard_enable(false) // disable all output from rasterizer
        .polygon_mode(vk::PolygonMode::FILL) // how fragments are generated: FILL, LINE, POINT
        .line_width(1.0) // thickness of lines in number of fragments, >1.0 requires GPU feature
        .cull_mode(vk::CullModeFlags::BACK) // what faces to remove
        // Due to Y-axis flip in projection matrix, we change from clockwise to counter clockwise.
        .front_face(vk::FrontFace::COUNTER_CLOCKWISE) // ordering of vertices for front side
        .depth_bias_enable(false); // add some value to depth values, used for shadow mapping

    // Use multi sample.
    let multisample_state = vk::PipelineMultisampleStateCreateInfo::builder()
        .sample_shading_enable(true) // enable in-texture anti-aliasing
        .min_sample_shading(0.2) // msaa_samples fraction of fragments calculated per texture pixel
        .rasterization_samples(data.msaa_samples);

    // The attachment state contains color blending configuration per attached framebuffer. This can
    // be used to define color mixing, e.g. alpha blending.
    let attachment = vk::PipelineColorBlendAttachmentState::builder()
        .color_write_mask(vk::ColorComponentFlags::all()) // determines which channels pass through
        .blend_enable(true) // Enable alpha blending.
        .src_color_blend_factor(vk::BlendFactor::SRC_ALPHA)
        .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
        .color_blend_op(vk::BlendOp::ADD)
        .src_alpha_blend_factor(vk::BlendFactor::ONE)
        .dst_alpha_blend_factor(vk::BlendFactor::ZERO)
        .alpha_blend_op(vk::BlendOp::ADD);

    // The global color blending settings that reference the array of structures for all
    // framebuffers. You can set blend constants here and define bitwise operation blending.
    // Enabling bitwise operations will disable color blending from the attachment structs, but
    // uses the color_write_mask parameter to determine the affected channels.
    let attachments = &[attachment];
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo::builder()
        .logic_op_enable(false)
        .logic_op(vk::LogicOp::COPY) // bitwise operation set here
        .attachments(attachments)
        .blend_constants([0.0, 0.0, 0.0, 0.0]);

    // Some parts of the pipeline can actually be changed without reconstructing the whole
    // pipeline, such as the viewport. To do this, you need to create dynamic state. Using dynamic
    // state, values put at pipeline creation are ignored and you need to provide the date at
    // drawing time.
    let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::LINE_WIDTH];
    let dynamic_state =
        vk::PipelineDynamicStateCreateInfo::builder().dynamic_states(dynamic_states);

    // We want to upload push constants.
    let vert_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::VERTEX)
        .offset(0) // values with no offset into push constants
        .size(size_of::<Mat4>() as u32); // our model matrix

    let frag_push_constant_range = vk::PushConstantRange::builder()
        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
        .offset(size_of::<Mat4>() as u32) // push constants are shared between all stages
        .size(size_of::<f32>() as u32); // our opacity value

    // Define the pipeline layout that is used to pass uniform variables to the shaders. We can
    // also specify push constants to pass on dynamic values.
    let set_layouts = &[data.descriptor_set_layout];
    let push_constant_ranges = &[vert_push_constant_range, frag_push_constant_range];
    let layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(set_layouts)
        .push_constant_ranges(push_constant_ranges);

    data.pipeline_layout = device.create_pipeline_layout(&layout_info, None)?;

    // Enable depth testing.
    let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::builder()
        .depth_test_enable(true)
        .depth_write_enable(true)
        .depth_compare_op(vk::CompareOp::LESS) // keep closer fragments
        // optional depth bound test (keep only values within depth range)
        .depth_bounds_test_enable(false)
        .min_depth_bounds(0.0)
        .max_depth_bounds(1.0)
        // optional stencil buffer operations, which needs a stencil component
        .stencil_test_enable(false)
        .front(vk::StencilOpState::builder().build())
        .back(vk::StencilOpState::builder().build());

    // With all information now set up, we can create the actual pipeline. You can use other
    // render passes with this pipeline but they need to be compatible with the render pass we
    // defined previously.
    let stages = &[vert_stage, frag_stage];
    let info = vk::GraphicsPipelineCreateInfo::builder()
        .stages(stages)
        .vertex_input_state(&vertex_input_state)
        .input_assembly_state(&input_assembly_state)
        .viewport_state(&viewport_state)
        .rasterization_state(&rasterization_state)
        .multisample_state(&multisample_state)
        .depth_stencil_state(&depth_stencil_state)
        .color_blend_state(&color_blend_state)
        .layout(data.pipeline_layout)
        .render_pass(data.render_pass)
        .subpass(0)
        // Below are optional arguments, which are used to derive a new pipeline from an existing
        // one. This eases creation if many functions are common between them and switching between
        // child and parent pipeline can also be quicker. These values are only used if you use the
        // DERIVATE flag.
        // .flags(vk::PipelineCreateFlags::DERIVATIVE)
        .base_pipeline_handle(vk::Pipeline::null())
        .base_pipeline_index(-1);

    // Vulkan can take multiple pipeline info structs to create multiple pipelines at once. The
    // first parameter is a cache that can store data for faster pipeline creation, which can also
    // be stored to and read from a file.
    data.pipeline = device
        .create_graphics_pipelines(vk::PipelineCache::null(), &[info], None)?
        .0[0];

    device.destroy_shader_module(vert_shader_module, None);
    device.destroy_shader_module(frag_shader_module, None);

    Ok(())
}

// While a render pass specifies which attachment to draw to, the framebuffer specifies what
// attachment role a specific image view takes on. The render pass and framebuffer together thus
// define the render target.
unsafe fn create_framebuffers(device: &Device, data: &mut VulkanData) -> Result<()> {
    // A framebuffer is associated with an image view and needs to be compatible with the render
    // pass. Compatibility roughly means using the same number and types of attachments.
    // Currently there is always only one draw operation running at a time, so we only need a
    // single depth buffer and a single multisample image. With truly multiple draw operations, you
    // will need to use multiple of these resources and swap them out in the specific framebuffer
    // used for the next draw operation (i.e. create as many resources as swapchain images).
    data.framebuffers = data
        .swapchain_image_views
        .iter()
        .map(|i| {
            // Only a single subpass runs because of our semaphores on the depth buffer.
            // First render to a multisample image and then resolve for the swapchain image.
            let attachments = &[data.color_image_view, data.depth_image_view, *i];
            let create_info = vk::FramebufferCreateInfo::builder()
                .render_pass(data.render_pass)
                .attachments(attachments)
                .width(data.swapchain_extent.width)
                .height(data.swapchain_extent.height)
                .layers(1);

            device.create_framebuffer(&create_info, None)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(())
}

unsafe fn create_command_pools(
    instance: &Instance,
    device: &Device,
    data: &mut VulkanData,
) -> Result<()> {
    // Use this command pool for all initialization on the graphics queue.
    data.command_pool = create_command_pool(instance, device, data)?;

    // Transient command pool for transfer commands as they are only done once, so the driver can
    // optimize these pools.
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let info = vk::CommandPoolCreateInfo::builder()
        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
        .queue_family_index(indices.transfer);

    data.transfer_command_pool = device.create_command_pool(&info, None)?;

    // Create a command pool for each frame to make it easy to reset all render commands.
    let num_images = data.swapchain_images.len();
    for _ in 0..num_images {
        let command_pool = create_command_pool(instance, device, data)?;
        data.command_pools.push(command_pool);
    }

    Ok(())
}

unsafe fn create_command_pool(
    instance: &Instance,
    device: &Device,
    data: &mut VulkanData,
) -> Result<vk::CommandPool> {
    // Create a command pool for rendering commands. Command pools can only hold command buffers
    // that are destined for a single type of queue, so here we use the graphics queue.
    // Options for Command Pool flags:
    // TRANSIENT: hint that command buffers are rerecorded very frequently
    // RESET_COMMAND_BUFFER: allow individual buffer to be rerecorded, not all together
    // PROTECTED: store command buffers in "protected" memory
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let info = vk::CommandPoolCreateInfo::builder()
        // These flags do not affect the correctness of the application, but correct use can make
        // the application more performant.
        // .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER) // allow rewriting command buffers
        .flags(vk::CommandPoolCreateFlags::TRANSIENT) // notify all command buffers get freed fast
        .queue_family_index(indices.graphics);

    let command_pool = device.create_command_pool(&info, None)?;

    Ok(command_pool)
}

unsafe fn create_command_buffers(device: &Device, data: &mut VulkanData) -> Result<()> {
    // Command buffers are created for command pools. They will be deallocated automatically with
    // their command pool. Level can be PRIMARY to directly submit them to the queue, or SECONDARY
    // to be referenced by some other primary command buffer.
    let num_images = data.swapchain_images.len();
    for image_index in 0..num_images {
        let allocate_info = vk::CommandBufferAllocateInfo::builder()
            .command_pool(data.command_pools[image_index])
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffer = device.allocate_command_buffers(&allocate_info)?[0];
        data.command_buffers.push(command_buffer);
    }

    // Initialize the outer vector of secondary command buffers.
    data.secondary_command_buffers = vec![vec![]; data.swapchain_images.len()];

    Ok(())
}

// Events sent to Vulkan are performed asynchronously, so we need to use some synchronization
// mechanism to issue dependent commands only one after the other. We have two options: Fences or
// Semaphores. Their difference is that state of fences can be accessed from within the program
// using wait_for_fences. Fences are used to synchronize the application with the rendering
// operation, while semaphores are used to synchronize operations within or across command queues.
unsafe fn create_sync_objects(device: &Device, data: &mut VulkanData) -> Result<()> {
    // Currently has no fields
    let semaphore_info = vk::SemaphoreCreateInfo::builder();
    // Initialize fences with signaled state.
    let fence_info = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);

    for _ in 0..MAX_FRAMES_IN_FLIGHT {
        data.image_available_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);
        data.render_finished_semaphores
            .push(device.create_semaphore(&semaphore_info, None)?);

        data.in_flight_fences
            .push(device.create_fence(&fence_info, None)?);
    }

    data.images_in_flight = data
        .swapchain_images
        .iter()
        .map(|_| vk::Fence::null())
        .collect();

    Ok(())
}

unsafe fn get_memory_type_index(
    instance: &Instance,
    data: &VulkanData,
    properties: vk::MemoryPropertyFlags,
    requirements: vk::MemoryRequirements,
) -> Result<u32> {
    // Query GPU for available memory types. Returned structure has .memory_types and
    // .memory_heaps. Memory heap can be dedicated VRAM or swap space in RAM.
    let memory = instance.get_physical_device_memory_properties(data.physical_device);
    (0..memory.memory_type_count)
        .find(|i| {
            // The memory should be suitable for the data and allow us e.g. to write to it.
            let suitable = (requirements.memory_type_bits & (1 << i)) != 0;
            let memory_type = memory.memory_types[*i as usize];
            suitable && memory_type.property_flags.contains(properties)
        })
        .ok_or_else(|| anyhow!("Failed to find suitable memory type."))
}

// With Vulkan, we also have to do memory management on the GPU.
unsafe fn create_vertex_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut VulkanData,
) -> Result<()> {
    // Use helper function instead to simplify this function.
    let size = (size_of::<Vertex>() * data.vertices.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    // Map buffer memory into CPU accessible memory space.
    let memory = device.map_memory(
        staging_buffer_memory,
        0,                           // offset
        size,                        // vk::WHOLE_SIZE to map until end of memory allocation
        vk::MemoryMapFlags::empty(), // no flags available yet
    )?;

    // Copy over our data and remove mapping. This is possible here because we use HOST_COHERENT to
    // make sure VRAM is consistent with CPU writes (which can lead to worse performance). Another
    // approach is to flush explicitly to make sure the data is actually there.
    // Use flush_mapped_memory_ranges after write and invalidate_mapped_memory_ranges before read.
    // Actually, these options only make the driver aware of our writes and still may not be
    // transfered. However, Vulkan guarantees it will be available upon the next call to
    // queue_submit.
    memcpy(data.vertices.as_ptr(), memory.cast(), data.vertices.len());
    device.unmap_memory(staging_buffer_memory);

    // Transfer data to a device local buffer that is more optimized for rendering.
    let (vertex_buffer, vertex_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.vertex_buffer = vertex_buffer;
    data.vertex_buffer_memory = vertex_buffer_memory;

    copy_buffer(device, data, staging_buffer, vertex_buffer, size)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn create_buffer(
    instance: &Instance,
    device: &Device,
    data: &VulkanData,
    size: vk::DeviceSize,
    usage: vk::BufferUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Buffer, vk::DeviceMemory)> {
    let indices = QueueFamilyIndices::get(instance, data, data.physical_device)?;
    let queue_family_indices = &[indices.graphics, indices.transfer];

    let buffer_info = vk::BufferCreateInfo::builder()
        .size(size)
        .usage(usage)
        // Using concurrent here because we use transfer queue.
        .sharing_mode(vk::SharingMode::CONCURRENT)
        // We need to specify which queues are sharing the resource at creation.
        .queue_family_indices(queue_family_indices);

    let buffer = device.create_buffer(&buffer_info, None)?;

    let requirements = device.get_buffer_memory_requirements(buffer);

    // In a real project, you do not want to do too many allocations because the device is often
    // very limited, e.g. the GTX 1080 only allows for 4096 concurrent allocations. Instead you
    // would allocate a large portion of memory on the device and then write your own allocator to
    // split that memory up for your app's use.
    let memory_info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            properties,
            requirements,
        )?);

    let buffer_memory = device.allocate_memory(&memory_info, None)?;

    device.bind_buffer_memory(buffer, buffer_memory, 0)?;

    Ok((buffer, buffer_memory))
}

unsafe fn copy_buffer(
    device: &Device,
    data: &VulkanData,
    source: vk::Buffer,
    destination: vk::Buffer,
    size: vk::DeviceSize,
) -> Result<()> {
    // All operations through a queue need to be submitted as part of a command pool, even copying
    // operations.
    let command_buffer = begin_single_time_command(device, data, data.transfer_command_pool)?;

    // Region consists of a source buffer offset, destination buffer offset and size. You can not
    // use WHOLE_SIZE here.
    let regions = vk::BufferCopy::builder().size(size);
    device.cmd_copy_buffer(command_buffer, source, destination, &[regions]);

    end_single_time_transfer_command(
        device,
        data,
        command_buffer,
        data.transfer_command_pool,
        data.transfer_queue,
    )?;
    Ok(())
}

// This function is very similar to create_vertex_buffer. In real applications it is recommended to
// store multiple buffers into a single Vulkan buffer, such as storing both the vertex buffer and
// index buffer together and using an offset to get to the data you need. This way data is more
// cache friendly on the GPU and some optimizations even use the same memory location for different
// resources if they are not needed simultaneously on a render pass. This is known as aliasing and
// some Vulkan functions allow to set flags to make this use explicit.
unsafe fn create_index_buffer(
    instance: &Instance,
    device: &Device,
    data: &mut VulkanData,
) -> Result<()> {
    // Use the same data type for size as you use for your indices.
    let size = (size_of::<u32>() * data.indices.len()) as u64;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;

    memcpy(data.indices.as_ptr(), memory.cast(), data.indices.len());
    device.unmap_memory(staging_buffer_memory);

    let (index_buffer, index_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::INDEX_BUFFER,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.index_buffer = index_buffer;
    data.index_buffer_memory = index_buffer_memory;

    copy_buffer(device, data, staging_buffer, index_buffer, size)?;

    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    Ok(())
}

unsafe fn create_descriptor_set_layout(device: &Device, data: &mut VulkanData) -> Result<()> {
    let ubo_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(0) // used also in shader
        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER) // bindings separate per type
        .descriptor_count(1) // if we have an array and how many elements of descriptors
        // We can have multiple transformations e.g. when working with skeletal animation.
        .stage_flags(vk::ShaderStageFlags::VERTEX); // shaders that reads this data

    // Texture samplers are accessed via descriptors just like a uniform buffer.
    let sampler_binding = vk::DescriptorSetLayoutBinding::builder()
        .binding(1)
        // With the combined image sampler descriptor we can access a texture through a sampler.
        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(1)
        .stage_flags(vk::ShaderStageFlags::FRAGMENT); // texture read in fragment shader

    let bindings = &[ubo_binding, sampler_binding];
    let info = vk::DescriptorSetLayoutCreateInfo::builder().bindings(bindings);

    data.descriptor_set_layout = device.create_descriptor_set_layout(&info, None)?;

    Ok(())
}

unsafe fn create_uniform_buffers(
    instance: &Instance,
    device: &Device,
    data: &mut VulkanData,
) -> Result<()> {
    data.uniform_buffers.clear();
    data.uniform_buffers_memory.clear();

    for _ in 0..data.swapchain_images.len() {
        let (uniform_buffer, uniform_buffer_memory) = create_buffer(
            instance,
            device,
            data,
            size_of::<UniformBufferObject>() as u64,
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        // We want to change the data of the uniform buffers every frame so we do not copy it again
        // for faster access, as the copy alone will take too long.
        data.uniform_buffers.push(uniform_buffer);
        data.uniform_buffers_memory.push(uniform_buffer_memory);
    }

    Ok(())
}

// Validation layers will not check for problems with pool size! Some devices may work around a pool
// sized too small while others fail with an error OUT_OF_POOL_MEMORY. It is no longer a strict
// requirement to allocate only so many descriptors of a type as you set with descriptor_count, it
// remains a best practice to stay within the limits you set. You can enable validation errors for
// this again using Best Practice Validation.
unsafe fn create_descriptor_pool(device: &Device, data: &mut VulkanData) -> Result<()> {
    // Create one descriptor for all images in flight.
    let ubo_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::UNIFORM_BUFFER) // allocate descriptor of this type
        .descriptor_count(data.swapchain_images.len() as u32); // at most this many from this type

    // Create one sampling descriptor for all images in flight.
    let sampler_size = vk::DescriptorPoolSize::builder()
        .type_(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
        .descriptor_count(data.swapchain_images.len() as u32);

    let pool_sizes = &[ubo_size, sampler_size];
    let info = vk::DescriptorPoolCreateInfo::builder()
        .pool_sizes(pool_sizes) // description of all types that can be allocated in this pool
        .max_sets(data.swapchain_images.len() as u32) // at most this many from any type
        .flags(vk::DescriptorPoolCreateFlags::empty()); // can enable to free individual descriptor

    data.descriptor_pool = device.create_descriptor_pool(&info, None)?;

    Ok(())
}

unsafe fn create_descriptor_sets(device: &Device, data: &mut VulkanData) -> Result<()> {
    let layouts = vec![data.descriptor_set_layout; data.swapchain_images.len()];
    let info = vk::DescriptorSetAllocateInfo::builder()
        .descriptor_pool(data.descriptor_pool)
        .set_layouts(&layouts);

    // Descriptor sets get automatically destroyed with their pool.
    data.descriptor_sets = device.allocate_descriptor_sets(&info)?;

    // Initialize descriptors.
    for i in 0..data.swapchain_images.len() {
        let info = vk::DescriptorBufferInfo::builder()
            .buffer(data.uniform_buffers[i])
            .offset(0) // all access to buffer through this descriptor uses this offset
            .range(size_of::<UniformBufferObject>() as u64); // size of descriptor update

        let buffer_info = &[info];
        let ubo_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i]) // set to update
            .dst_binding(0) // set binding to update
            .dst_array_element(0) // descriptor index if using an array
            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER) // type at array location
            // Types of descriptors, only one per call is allowed:
            // .image_info(image_info)
            // .texel_buffer_view(texel_buffer_view)
            .buffer_info(buffer_info); // how many descriptors to update in a row

        // Same procedure with the texture sampler descriptor.
        let info = vk::DescriptorImageInfo::builder()
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .image_view(data.texture_image_view)
            .sampler(data.texture_sampler);

        let image_info = &[info];
        let sampler_write = vk::WriteDescriptorSet::builder()
            .dst_set(data.descriptor_sets[i])
            .dst_binding(1)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .image_info(image_info);

        // Update can override or copy a descriptor.
        device.update_descriptor_sets(&[ubo_write, sampler_write], &[] as &[vk::CopyDescriptorSet]);
    }

    Ok(())
}

unsafe fn create_image(
    instance: &Instance,
    device: &Device,
    data: &VulkanData,
    width: u32,
    height: u32,
    mip_levels: u32,
    samples: vk::SampleCountFlags,
    format: vk::Format,
    tiling: vk::ImageTiling,
    usage: vk::ImageUsageFlags,
    properties: vk::MemoryPropertyFlags,
) -> Result<(vk::Image, vk::DeviceMemory)> {
    let info = vk::ImageCreateInfo::builder()
        .image_type(vk::ImageType::_2D) // specify coordinate system of texels
        .extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        }) // layout of data
        .mip_levels(mip_levels)
        .array_layers(1) // only one image, not an array
        .format(format) // pixel layout, best to use the same as pixel output
        .tiling(tiling) // LINEAR (row-major) or OPTIMAL (implementation defined)
        .initial_layout(vk::ImageLayout::UNDEFINED) // UNDEFINED or PREINITIALIZED
        .usage(usage) // sample texture
        .sharing_mode(vk::SharingMode::EXCLUSIVE) // only used in graphics queue
        .samples(samples) // multisampling option when using image as attachment
        .flags(vk::ImageCreateFlags::empty()); // e.g. specify sparse textures

    let image = device.create_image(&info, None)?;

    // allocation very similar to allocating buffer memory
    let requirements = device.get_image_memory_requirements(image);
    let info = vk::MemoryAllocateInfo::builder()
        .allocation_size(requirements.size)
        .memory_type_index(get_memory_type_index(
            instance,
            data,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
            requirements,
        )?);
    let image_memory = device.allocate_memory(&info, None)?;
    // allocate image memory instead of buffer memory
    device.bind_image_memory(image, image_memory, 0)?;

    Ok((image, image_memory))
}

unsafe fn create_texture_image(
    instance: &Instance,
    device: &Device,
    data: &mut VulkanData,
) -> Result<()> {
    // Image is expected to have an alpha channel: RGBA
    // let image = File::open("resources/texture.png")?;
    let image = File::open("./resources/viking_room.png")?;

    let decoder = png::Decoder::new(image);
    let mut reader = decoder.read_info()?;

    let mut pixels = vec![0; reader.info().raw_bytes()];
    reader.next_frame(&mut pixels)?;

    let size = reader.info().raw_bytes() as u64;
    let (width, height) = reader.info().size();

    // calculate mip levels based on texture dimensions
    data.mip_levels = (width.max(height) as f32).log2().floor() as u32 + 1;

    let (staging_buffer, staging_buffer_memory) = create_buffer(
        instance,
        device,
        data,
        size,
        vk::BufferUsageFlags::TRANSFER_SRC,
        vk::MemoryPropertyFlags::HOST_COHERENT | vk::MemoryPropertyFlags::HOST_VISIBLE,
    )?;

    let memory = device.map_memory(staging_buffer_memory, 0, size, vk::MemoryMapFlags::empty())?;
    memcpy(pixels.as_ptr(), memory.cast(), pixels.len());
    device.unmap_memory(staging_buffer_memory);

    let (texture_image, texture_image_memory) = create_image(
        instance,
        device,
        data,
        width,
        height,
        data.mip_levels,
        vk::SampleCountFlags::_1,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::TRANSFER_SRC, // creating mipmap is considered a transfer
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.texture_image = texture_image;
    data.texture_image_memory = texture_image_memory;

    // Make sure the image has its data storage set correctly. All these helper functions wait for
    // the queue to be idle, which means we are synchronous with the device. You would want to
    // change this to asynchronous calls for more efficiency, e.g. by combining all these commands
    // to a single command buffer.
    transition_image_layout(
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        data.mip_levels,
        data.command_pool,
        data.graphics_queue,
    )?;

    copy_buffer_to_image(
        device,
        data,
        staging_buffer,
        data.texture_image,
        width,
        height,
    )?;

    // Destroy staging buffer here.
    device.destroy_buffer(staging_buffer, None);
    device.free_memory(staging_buffer_memory, None);

    // Keep all images in Transfer Dst optimal layout for mipmap creation.
    generate_mipmaps(
        instance,
        device,
        data,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        width,
        height,
        data.mip_levels,
    )?;

    Ok(())
}

unsafe fn begin_single_time_command(
    device: &Device,
    data: &VulkanData,
    command_pool: vk::CommandPool,
) -> Result<vk::CommandBuffer> {
    let info = vk::CommandBufferAllocateInfo::builder()
        .level(vk::CommandBufferLevel::PRIMARY)
        .command_pool(command_pool)
        .command_buffer_count(1);

    let command_buffer = device.allocate_command_buffers(&info)?[0];

    // Make our intent clear to the driver with ONE_TIME_SUBMIT.
    let info =
        vk::CommandBufferBeginInfo::builder().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

    device.begin_command_buffer(command_buffer, &info)?;

    Ok(command_buffer)
}

unsafe fn end_single_time_transfer_command(
    device: &Device,
    data: &VulkanData,
    command_buffer: vk::CommandBuffer,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<()> {
    device.end_command_buffer(command_buffer)?;

    let command_buffers = &[command_buffer];
    let info = vk::SubmitInfo::builder().command_buffers(command_buffers);

    // Submit copy command to transfer queue and wait until it is completed. You could also wait
    // for a fence that specifically marks this command complete.
    device.queue_submit(queue, &[info], vk::Fence::null())?;
    device.queue_wait_idle(queue)?;

    device.free_command_buffers(command_pool, &[command_buffer]);

    Ok(())
}

unsafe fn transition_image_layout(
    device: &Device,
    data: &VulkanData,
    image: vk::Image,
    format: vk::Format,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
    mip_levels: u32,
    command_pool: vk::CommandPool,
    queue: vk::Queue,
) -> Result<()> {
    let command_buffer = begin_single_time_command(device, data, command_pool)?;

    // Make this function also work for depth formats.
    let aspect_mask = if new_layout == vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL {
        match format {
            vk::Format::D32_SFLOAT_S8_UINT | vk::Format::D24_UNORM_S8_UINT => {
                vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
            }
            _ => vk::ImageAspectFlags::DEPTH,
        }
    } else {
        vk::ImageAspectFlags::COLOR
    };

    // Specify the affected image and the part of it.
    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspect_mask)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    // This function is only called with these combinations. Command buffer submissions also
    // includes an implicit HOST_WRITE access flag if no other flags are set.
    let (src_access_mask, dst_access_mask, src_stage_mask, dst_stage_mask) =
        match (old_layout, new_layout) {
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::TRANSFER_DST_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER, // pseudo stage for transfer
            ),
            (vk::ImageLayout::TRANSFER_DST_OPTIMAL, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL) => (
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
            ),
            (vk::ImageLayout::UNDEFINED, vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL) => (
                vk::AccessFlags::empty(),
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                // Reading is done in early fragment test, writing in late fragment test.
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            ),
            _ => return Err(anyhow!("Unsupported image layout transition!")),
        };

    // Pipeline barriers are used to synchronize access to resources. A layout transition is
    // integrated into a memory barrier because we need to guarantee anyway that all writes are
    // done before the transition and any reads are done only after the transition.
    let barrier = vk::ImageMemoryBarrier::builder()
        .old_layout(old_layout)
        .new_layout(new_layout)
        // These are used to transfer ownership if using EXCLUSIVE share mode. IGNORED is not the
        // default value!
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(subresource)
        // Specify the operations that need to be done before and which need to wait. This must be
        // done in addition to queue_wait_idle. TODO
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask);

    // Submit pipeline barrier.
    device.cmd_pipeline_barrier(
        command_buffer,
        // Available access flags depend on the stage. Validation layers will warn about unlogical
        // combinations. TODO
        src_stage_mask,                    // commands first to happen in this stage
        dst_stage_mask,                    // commands waiting in this stage
        vk::DependencyFlags::empty(),      // BY_REGION: barrier acts per region that is completed
        &[] as &[vk::MemoryBarrier],       // memory barriers
        &[] as &[vk::BufferMemoryBarrier], // buffer memory barriers
        &[barrier],                        // image memory barriers
    );

    end_single_time_transfer_command(device, data, command_buffer, command_pool, queue)?;
    Ok(())
}

unsafe fn copy_buffer_to_image(
    device: &Device,
    data: &VulkanData,
    buffer: vk::Buffer,
    image: vk::Image,
    width: u32,
    height: u32,
) -> Result<()> {
    let command_buffer = begin_single_time_command(device, data, data.command_pool)?;

    let subresource = vk::ImageSubresourceLayers::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .base_array_layer(0)
        .layer_count(1);

    let region = vk::BufferImageCopy::builder()
        .buffer_offset(0)
        // 0 length indicates tightly packed rows and columns
        .buffer_row_length(0)
        .buffer_image_height(0)
        .image_subresource(subresource)
        .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
        .image_extent(vk::Extent3D {
            width,
            height,
            depth: 1,
        });

    device.cmd_copy_buffer_to_image(
        command_buffer,
        buffer,
        image,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL, // current layout of image
        &[region],                             // can specify multiple regions to copy over at once
    );

    end_single_time_transfer_command(
        device,
        data,
        command_buffer,
        data.command_pool,
        data.graphics_queue,
    )?;
    Ok(())
}

unsafe fn create_image_view(
    device: &Device,
    image: vk::Image,
    format: vk::Format,
    aspects: vk::ImageAspectFlags,
    mip_levels: u32,
) -> Result<vk::ImageView> {
    // Define a color component mapping, which allows remapping color channels to one another, etc.
    let components = vk::ComponentMapping::builder()
        .r(vk::ComponentSwizzle::IDENTITY)
        .g(vk::ComponentSwizzle::IDENTITY)
        .b(vk::ComponentSwizzle::IDENTITY)
        .a(vk::ComponentSwizzle::IDENTITY);

    // Define image purpose and the part of the image we want to access.
    // You can choose different layers when you use a corresponding swapchain.
    let subresource_range = vk::ImageSubresourceRange::builder()
        .aspect_mask(aspects)
        .base_mip_level(0)
        .level_count(mip_levels)
        .base_array_layer(0)
        .layer_count(1);

    let info = vk::ImageViewCreateInfo::builder()
        .image(image)
        .view_type(vk::ImageViewType::_2D)
        .format(format)
        .components(components)
        .subresource_range(subresource_range);

    let image_view = device.create_image_view(&info, None)?;

    Ok(image_view)
}

unsafe fn create_texture_image_view(device: &Device, data: &mut VulkanData) -> Result<()> {
    // Images are never directly accessed, which is also true for texture images.
    data.texture_image_view = create_image_view(
        device,
        data.texture_image,
        vk::Format::R8G8B8A8_SRGB,
        vk::ImageAspectFlags::COLOR,
        data.mip_levels,
    )?;

    Ok(())
}

unsafe fn create_texture_sampler(device: &Device, data: &mut VulkanData) -> Result<()> {
    let info = vk::SamplerCreateInfo::builder()
        .mag_filter(vk::Filter::LINEAR) // magnifying filter
        .min_filter(vk::Filter::LINEAR) // minifying filter
        // Addressing mode for each coordinate system axis, useful for tiling.
        // REPEAT, MIRRORED_REPEAT, CLAMP_TO_EDGE, MIRROR_CLAMP_TO_EDGE, CLAMP_TO_BORDER
        .address_mode_u(vk::SamplerAddressMode::REPEAT)
        .address_mode_v(vk::SamplerAddressMode::REPEAT)
        .address_mode_w(vk::SamplerAddressMode::REPEAT)
        .anisotropy_enable(true) // helps for sharp angles
        .max_anisotropy(16.0) // 16 is maximum support on hardware because result become stale
        .border_color(vk::BorderColor::INT_OPAQUE_BLACK) // used with CLAMP_TO_BORDER
        .unnormalized_coordinates(false) // sample between [0, 1) instead of [0, width) etc.
        .compare_enable(false) // enable comparing value with an operation
        .compare_op(vk::CompareOp::ALWAYS) // comparison performed before going to filter
        // Options for mipmapping.
        .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
        .mip_lod_bias(0.0)
        .min_lod(0.0)
        // .min_lod(data.mip_levels as f32 / 2.0)
        .max_lod(data.mip_levels as f32);

    // Sampler are distinct objects from image views, so you can reuse the same sampler on
    // different textures.
    data.texture_sampler = device.create_sampler(&info, None)?;

    Ok(())
}

unsafe fn get_supported_format(
    instance: &Instance,
    data: &VulkanData,
    candidates: &[vk::Format],
    tiling: vk::ImageTiling,
    features: vk::FormatFeatureFlags,
) -> Result<vk::Format> {
    candidates
        .iter()
        .find(|f| {
            let properties =
                instance.get_physical_device_format_properties(data.physical_device, **f);

            match tiling {
                vk::ImageTiling::LINEAR => properties.linear_tiling_features.contains(features),
                vk::ImageTiling::OPTIMAL => properties.optimal_tiling_features.contains(features),
                _ => false,
            }
        })
        .cloned()
        .ok_or_else(|| anyhow!("Failed to find supported format!"))
}

unsafe fn get_depth_format(instance: &Instance, data: &VulkanData) -> Result<vk::Format> {
    let candidates = &[
        vk::Format::D32_SFLOAT,
        vk::Format::D32_SFLOAT_S8_UINT,
        vk::Format::D24_UNORM_S8_UINT,
    ];

    get_supported_format(
        instance,
        data,
        candidates,
        vk::ImageTiling::OPTIMAL,
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT,
    )
}

unsafe fn create_depth_object(
    instance: &Instance,
    device: &Device,
    data: &mut VulkanData,
) -> Result<()> {
    // Depth values will not be accessed directly by our program.
    let format = get_depth_format(instance, data)?;

    let (depth_image, depth_image_memory) = create_image(
        instance,
        device,
        data,
        data.swapchain_extent.width,
        data.swapchain_extent.height,
        1,                 // depth image has no mipmap
        data.msaa_samples, // but all samples have depth
        format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.depth_image = depth_image;
    data.depth_image_memory = depth_image_memory;

    data.depth_image_view = create_image_view(
        device,
        depth_image,
        format,
        vk::ImageAspectFlags::DEPTH,
        1, // no mipmap
    )?;

    // Optional transition of image layout,
    transition_image_layout(
        device,
        data,
        data.depth_image,
        format,
        vk::ImageLayout::UNDEFINED,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        1, // no mipmap
        data.command_pool,
        data.graphics_queue,
    )?;

    Ok(())
}

fn load_model(data: &mut VulkanData) -> Result<()> {
    let mut reader = BufReader::new(File::open("./resources/viking_room.obj")?);
    let (models, _) = tobj::load_obj_buf(
        &mut reader,
        &tobj::LoadOptions {
            triangulate: true, // convert all primitives to triangles
            single_index: true,
            ..Default::default()
        },
        |_| Ok(Default::default()), // ignore materials
    )?;

    let mut unique_vertices = HashMap::new();
    for model in &models {
        data.vertices.clear();
        data.indices.clear();

        for index in &model.mesh.indices {
            let pos_offset = (3 * index) as usize;
            let tex_coord_offset = (2 * index) as usize;

            let vertex = Vertex {
                pos: vec3(
                    model.mesh.positions[pos_offset],
                    model.mesh.positions[pos_offset + 1],
                    model.mesh.positions[pos_offset + 2],
                ),
                color: vec3(1.0, 1.0, 1.0),
                tex_coord: vec2(
                    model.mesh.texcoords[tex_coord_offset],
                    1.0 - (model.mesh.texcoords[tex_coord_offset + 1]),
                ),
            };

            if let Some(index) = unique_vertices.get(&vertex) {
                data.indices.push(*index as u32)
            } else {
                let index = data.vertices.len();
                unique_vertices.insert(vertex, index);
                data.vertices.push(vertex);
                data.indices.push(index as u32);
            }
        }
    }

    Ok(())
}

unsafe fn generate_mipmaps(
    instance: &Instance,
    device: &Device,
    data: &VulkanData,
    image: vk::Image,
    format: vk::Format,
    width: u32,
    height: u32,
    mip_levels: u32,
) -> Result<()> {
    if !instance
        .get_physical_device_format_properties(data.physical_device, format)
        .optimal_tiling_features // we use optimal tiling format
        .contains(vk::FormatFeatureFlags::SAMPLED_IMAGE_FILTER_LINEAR)
    {
        // Alternatives if this is not available:
        // 1. Search for a common texture image format where your device supports linear filtering.
        // 2. Implement the mipmap generation on the CPU side.
        // Mipmap generation is mostly done offline anyway.
        return Err(anyhow!(
            "Texture image format does not support linear blitting!"
        ));
    }

    // Blit is done through the graphics queue.
    let command_buffer = begin_single_time_command(device, data, data.command_pool)?;

    let mut mip_width = width;
    let mut mip_height = height;
    for i in 1..mip_levels {
        // Make level i - 1 ready to send data. All layers are DST beforehand.
        let subresource = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(1)
            .base_mip_level(i - 1)
            .level_count(1); // can only see this mip level

        let barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(subresource);

        device.cmd_pipeline_barrier(
            command_buffer,
            // Wait until we are at the transfer stage.
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        let src_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i - 1)
            .base_array_layer(0)
            .layer_count(1);

        let dst_subresource = vk::ImageSubresourceLayers::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(i)
            .base_array_layer(0)
            .layer_count(1);

        // Blit operations use a source and destination bitmap and performs scaling and filtering.
        // These can also be used to create mipmaps.
        let blit = vk::ImageBlit::builder()
            .src_offsets([
                // determine rectangle area to blit
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: mip_width as i32,
                    y: mip_height as i32,
                    z: 1,
                },
            ])
            .src_subresource(src_subresource) // layer i - 1
            .dst_offsets([
                // write data in rectangle area
                vk::Offset3D { x: 0, y: 0, z: 0 },
                vk::Offset3D {
                    x: (if mip_width > 1 { mip_width / 2 } else { 1 }) as i32,
                    y: (if mip_height > 1 { mip_height / 2 } else { 1 }) as i32,
                    z: 1,
                },
            ])
            .dst_subresource(dst_subresource); // layer i

        device.cmd_blit_image(
            command_buffer,
            image, // image as src and dst
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[blit],
            vk::Filter::LINEAR,
        );

        let subresource = vk::ImageSubresourceRange::builder()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .base_array_layer(0)
            .layer_count(1)
            .base_mip_level(i - 1)
            .level_count(1); // can only see this mip level

        let barrier = vk::ImageMemoryBarrier::builder()
            .image(image)
            .src_access_mask(vk::AccessFlags::TRANSFER_READ)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(subresource);

        // Make sure all data is written before it is used in the shader.
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[] as &[vk::MemoryBarrier],
            &[] as &[vk::BufferMemoryBarrier],
            &[barrier],
        );

        if mip_width > 1 {
            mip_width /= 2;
        }
        if mip_height > 1 {
            mip_height /= 2;
        }
    }

    let subresource = vk::ImageSubresourceRange::builder()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .base_array_layer(0)
        .layer_count(1)
        .base_mip_level(mip_levels - 1)
        .level_count(1); // can only see this mip level

    let barrier = vk::ImageMemoryBarrier::builder()
        .image(image)
        .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
        .dst_access_mask(vk::AccessFlags::SHADER_READ)
        .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .subresource_range(subresource);

    // Transition last layer too.
    device.cmd_pipeline_barrier(
        command_buffer,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::FRAGMENT_SHADER,
        vk::DependencyFlags::empty(),
        &[] as &[vk::MemoryBarrier],
        &[] as &[vk::BufferMemoryBarrier],
        &[barrier],
    );

    end_single_time_transfer_command(
        device,
        data,
        command_buffer,
        data.command_pool,
        data.graphics_queue,
    )?;

    Ok(())
}

unsafe fn get_max_msaa_samples(instance: &Instance, data: &VulkanData) -> vk::SampleCountFlags {
    let properties = instance.get_physical_device_properties(data.physical_device);
    let counts = properties.limits.framebuffer_color_sample_counts
        & properties.limits.framebuffer_depth_sample_counts;

    [
        vk::SampleCountFlags::_64,
        vk::SampleCountFlags::_32,
        vk::SampleCountFlags::_16,
        vk::SampleCountFlags::_8,
        vk::SampleCountFlags::_4,
        vk::SampleCountFlags::_2,
    ]
    .iter()
    .find(|c| counts.contains(**c))
    .cloned()
    .unwrap_or(vk::SampleCountFlags::_1)
}
unsafe fn create_color_objects(
    instance: &Instance,
    device: &Device,
    data: &mut VulkanData,
) -> Result<()> {
    let (color_image, color_image_memory) = create_image(
        instance,
        device,
        data,
        data.swapchain_extent.width,
        data.swapchain_extent.height,
        1,                 // With multiple samples, the spec enforces only 1 mip level in case
        data.msaa_samples, // of multi-sample images. It is no texture anyway.
        data.swapchain_format,
        vk::ImageTiling::OPTIMAL,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSIENT_ATTACHMENT,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;

    data.color_image = color_image;
    data.color_image_memory = color_image_memory;

    data.color_image_view = create_image_view(
        device,
        data.color_image,
        data.swapchain_format,
        vk::ImageAspectFlags::COLOR,
        1,
    )?;

    Ok(())
}
