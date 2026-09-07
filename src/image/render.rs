//! Orchestration to render the image viewer widget

use std::range::Range;

use iced::wgpu;
use nalgebra as na;

use crate::controls::RenderContext;
use crate::image::{DrawParameters, Image};
use crate::instruments::bind::image::{LanczosInfoRaw, ViewportRaw};
use crate::instruments::bind::storage::{
    SimpleStorageTexture,
    StorageSrcDstLayout,
    StorageTextureCopyMachine,
};
use crate::instruments::bind::texture::{SimpleTexture, SimpleTextureLayout};
use crate::instruments::buffer::{
    SimpleBuffer,
    SimpleBufferBind,
    SimpleBufferBindLayout,
    VisibleFragment,
    VisibleVertex,
};
use crate::instruments::mesh::InstanceBuffer;
use crate::instruments::mesh::primitives::InstanceRaw;
use crate::instruments::mesh::quad::QuadMesh;
use crate::instruments::pipeline::filter::{
    ConvolutionPipeline,
    ConvolutionPipelineLayout,
    KernelBinding,
    KernelLayout,
};
use crate::instruments::pipeline::image::{
    LanczosImageRenderPipelineLayout,
    RenderBilinearPipeline,
    RenderLanczosPipeline,
    RenderNearestPipeline,
    SimpleImageRenderPipelineLayout,
};
use crate::instruments::pipeline::passthru::{PassThruPipeline, PassThruTexture};
use crate::instruments::{GpuContext, TargetContext, Use};
use crate::viewport::ViewportGui;

/// The rendering instruments independent of any concrete image
#[derive(Default)]
pub struct ImageMetaInstruments {
    /// The current complete rendering
    final_output: Use<PassThruTexture>,

    /// The storage layout used during bluring
    storage_layout: Use<StorageSrcDstLayout>,
    /// The blur kernel layout
    kernel_layout: Use<KernelLayout>,
    /// The convolution pipeline layout for blurring
    convolution_layout: Use<ConvolutionPipelineLayout>,
    /// The convolution pipeline for blurring
    convolution_pipeline: Use<ConvolutionPipeline>,
    /// The copy machine used during bluring
    copy_machine: Use<StorageTextureCopyMachine>,

    /// The universal texture layout
    texture_layout: Use<SimpleTextureLayout>,
    /// The vertex buffer bind group layout
    buffer_layout_vertex: Use<SimpleBufferBindLayout<VisibleVertex>>,
    /// The fragment buffer bind group layout
    buffer_layout_fragment: Use<SimpleBufferBindLayout<VisibleFragment>>,

    /// The simple pipeline layout
    simple_pipeline_layout: Use<SimpleImageRenderPipelineLayout>,
    /// The lanczos pipeline layout
    lanczos_pipeline_layout: Use<LanczosImageRenderPipelineLayout>,
    /// The pipeline to render with "nearest" filter
    nearest_pipeline: Use<RenderNearestPipeline>,
    /// The pipeline to render with "bilinear" filter
    bilinear_pipeline: Use<RenderBilinearPipeline>,
    /// The pipeline to render with "lanczos" filter
    lanczos_pipeline: Use<RenderLanczosPipeline>,

    /// A quad mesh that provides the base vertices during rendering
    quad_mesh: Use<QuadMesh>,
    /// The buffer for image metadata used in all render pipelines
    viewport_buffer: Use<SimpleBufferBind<ViewportRaw, VisibleVertex>>,
    /// The buffer for lanczos filter metadata
    lanczos_buffer: Use<SimpleBufferBind<LanczosInfoRaw, VisibleFragment>>,
}

/// Public API
impl ImageMetaInstruments {
    /// Create a new set of meta image instruments
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the current final output
    pub const fn final_output(&self) -> Option<&PassThruTexture> {
        let Use::Active(output) = &self.final_output else {
            return None;
        };
        Some(output)
    }

    /// Update the current final output with a new result
    pub fn set_final_output(&mut self, output: PassThruTexture) -> &PassThruTexture {
        self.final_output = Use::Active(output);
        self.final_output.active()
    }

    /// Render a new image with the provided [`ImageDataInstruments`] using "nearest" filter
    pub fn nearest(
        &mut self,
        data: &mut ImageDataInstruments,
        context: &mut RenderContext,
        image: &Image,
        params: &DrawParameters,
    ) {
        let RenderContext {
            ctx,
            target,
            passthru,
            encoder,
            viewport,
        } = context;

        let Some(output) = data.create_output(ctx, viewport, passthru) else {
            return;
        };

        self.create_quad_mesh(ctx);
        self.create_original(data, ctx, image);
        self.create_instance(data, ctx, image);
        self.create_nearest_pipeline(ctx, target);
        self.create_viewport_buffer(ctx, params);

        let quad_mesh = self.quad_mesh.active();
        let original = data.original.active();
        let instance = data.instance.active();
        let pipeline = self.nearest_pipeline.active();
        let viewport = self.viewport_buffer.active();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Nearest Image Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output.view(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pipeline.draw(&mut pass, viewport, original, quad_mesh, instance);

        data.output = Use::Active(output);
    }

    /// Render a new image with the provided [`ImageDataInstruments`] using "bilinear" filter
    pub fn bilinear(
        &mut self,
        data: &mut ImageDataInstruments,
        context: &mut RenderContext,
        image: &Image,
        params: &DrawParameters,
    ) {
        let RenderContext {
            ctx,
            target,
            passthru,
            encoder,
            viewport,
        } = context;

        let Some(output) = data.create_output(ctx, viewport, passthru) else {
            return;
        };

        self.create_quad_mesh(ctx);
        self.create_original(data, ctx, image);
        self.create_instance(data, ctx, image);
        self.create_bilinear_pipeline(ctx, target);
        self.create_viewport_buffer(ctx, params);

        let quad_mesh = self.quad_mesh.active();
        let original = data.original.active();
        let instance = data.instance.active();
        let pipeline = self.bilinear_pipeline.active();
        let viewport = self.viewport_buffer.active();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Bilinear Image Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output.view(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pipeline.draw(&mut pass, viewport, original, quad_mesh, instance);

        data.output = Use::Active(output);
    }

    /// Render a new image with the provided [`ImageDataInstruments`] using "lanczos" filter
    pub fn lanczos(
        &mut self,
        data: &mut ImageDataInstruments,
        context: &mut RenderContext,
        image: &Image,
        params: &DrawParameters,
    ) {
        let RenderContext {
            ctx,
            target,
            passthru,
            encoder,
            viewport,
        } = context;

        let Some(output) = data.create_output(ctx, viewport, passthru) else {
            return;
        };

        // Interpolate with Lanczos filter
        self.create_quad_mesh(ctx);
        self.create_instance(data, ctx, image);
        self.create_viewport_buffer(ctx, params);

        self.create_blurred(data, ctx, encoder, image, params);
        self.create_lanczos_pipeline(ctx, target);
        self.create_lanczos_buffer(ctx, params);

        let blurred = if let Some(blurred) = &data.blurred.maybe_active() {
            blurred
        } else {
            self.create_original(data, ctx, image);
            data.original.active()
        };

        let quad_mesh = self.quad_mesh.active();
        let instance = data.instance.active();
        let pipeline = self.lanczos_pipeline.active();
        let lanczos_buffer = self.lanczos_buffer.active();
        let viewport = self.viewport_buffer.active();

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Lanczos Image Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: output.view(),
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pipeline.draw(
            &mut pass,
            viewport,
            blurred,
            lanczos_buffer,
            quad_mesh,
            instance,
        );

        data.output = Use::Active(output);
    }
}

/// Instrumentalization
impl ImageMetaInstruments {
    /// Extract or create the render target for the widget
    pub fn create_output(
        &mut self,
        ctx: &GpuContext,
        passthru: &PassThruPipeline,
        viewport: &ViewportGui,
    ) -> Option<PassThruTexture> {
        let Some(extent) = viewport.extent() else {
            self.final_output = self.final_output.take().make_unused();
            return None;
        };

        match self.final_output.take() {
            Use::Invalid => {
                self.final_output = Use::Invalid;
                None
            }
            Use::Active(output) | Use::Recycle(output) | Use::Unused(output)
                if output.texture().size() == extent =>
            {
                Some(output)
            }
            Use::Active(_) | Use::Recycle(_) | Use::Unused(_) | Use::Missing => {
                Some(passthru.create_texture(ctx, extent))
            }
        }
    }

    /// Make sure the texture layout is available
    fn create_texture_layout(&mut self, ctx: &GpuContext) {
        if self.texture_layout.checked() {
            return;
        }

        let out = SimpleTextureLayout::new(ctx, Some("Image Texture Layout"));
        self.texture_layout = Use::Active(out);
    }

    /// Make sure the buffer layout is available
    fn create_buffer_layout_vertex(&mut self, ctx: &GpuContext) {
        if self.buffer_layout_vertex.checked() {
            return;
        }
        let out = SimpleBufferBindLayout::new(ctx, Some("Image Buffer Layout Vertex"));
        self.buffer_layout_vertex = Use::Active(out);
    }

    /// Make sure the buffer layout is available
    fn create_buffer_layout_fragment(&mut self, ctx: &GpuContext) {
        if self.buffer_layout_fragment.checked() {
            return;
        }
        let out = SimpleBufferBindLayout::new(ctx, Some("Image Buffer Layout Fragment"));
        self.buffer_layout_fragment = Use::Active(out);
    }

    /// Make sure the simple pipeline layout is available
    fn create_simple_pipeline_layout(&mut self, ctx: &GpuContext) {
        if self.simple_pipeline_layout.checked() {
            return;
        }

        self.create_texture_layout(ctx);
        self.create_buffer_layout_vertex(ctx);

        let texture = self.texture_layout.active();
        let buffer = self.buffer_layout_vertex.active();

        let pipeline = SimpleImageRenderPipelineLayout::new(ctx, buffer, texture);
        self.simple_pipeline_layout = Use::Active(pipeline);
    }

    /// Make sure the pipeline for the "nearest" filter is available
    fn create_nearest_pipeline(&mut self, ctx: &GpuContext, target: &TargetContext) {
        if self.nearest_pipeline.checked() {
            return;
        }

        self.create_simple_pipeline_layout(ctx);
        let pipeline = self.simple_pipeline_layout.active();

        let pipeline = RenderNearestPipeline::new(ctx, pipeline, target.config.format);
        self.nearest_pipeline = Use::Active(pipeline);
    }

    /// Make sure the pipeline for the "bilinear" filter is available
    fn create_bilinear_pipeline(&mut self, ctx: &GpuContext, target: &TargetContext) {
        if self.bilinear_pipeline.checked() {
            return;
        }

        self.create_simple_pipeline_layout(ctx);
        let pipeline = self.simple_pipeline_layout.active();

        let pipeline = RenderBilinearPipeline::new(ctx, pipeline, target.config.format);
        self.bilinear_pipeline = Use::Active(pipeline);
    }

    /// Make sure image metadata buffer is available
    fn create_viewport_buffer(&mut self, ctx: &GpuContext, params: &DrawParameters) {
        match self.viewport_buffer.take() {
            Use::Missing => {
                self.create_buffer_layout_vertex(ctx);
                let layout = self.buffer_layout_vertex.active();

                let viewport = params.viewport.as_raw();
                let buffer = SimpleBuffer::new(ctx, viewport, Some("Image Viewport Buffer"));
                let bind =
                    SimpleBufferBind::new(ctx, buffer, layout, Some("Image Viewport Binding"));
                self.viewport_buffer = Use::Active(bind);
            }
            Use::Recycle(bind) | Use::Unused(bind) => {
                let viewport = params.viewport.as_raw();
                bind.buffer().update(ctx, viewport);
                self.viewport_buffer = Use::Active(bind);
            }
            out @ (Use::Invalid | Use::Active(_)) => {
                self.viewport_buffer = out;
            }
        }
    }

    /// Make sure the original image has been uploaded to a texture
    fn create_original(
        &mut self,
        data: &mut ImageDataInstruments,
        ctx: &GpuContext,
        image: &Image,
    ) {
        if data.original.checked() {
            return;
        }

        self.create_texture_layout(ctx);
        let texture_layout = self.texture_layout.active();
        let texture = image.upload(ctx, Some("Image Original Texture"));
        let original = SimpleTexture::new(
            ctx,
            texture_layout,
            texture,
            Some("Image Original Texture Bind"),
        );
        data.original = Use::Active(original);
    }

    /// Create the quad mesh (which never changes again)
    fn create_quad_mesh(&mut self, ctx: &GpuContext) {
        if self.quad_mesh.checked() {
            return;
        }

        self.quad_mesh = Use::Active(QuadMesh::new(ctx));
    }

    /// Create the quad instance that transforms the quad mesh to the image size
    #[expect(clippy::unused_self)]
    fn create_instance(&self, data: &mut ImageDataInstruments, ctx: &GpuContext, image: &Image) {
        if data.instance.checked() {
            return;
        }

        let wgpu::Extent3d { width, height, .. } = image.extent();
        let size = na::Vector2::new(width, height).cast();
        let instance = QuadMesh::box_instance(size);
        let instance = InstanceBuffer::upload(ctx, &instance);
        data.instance = Use::Active(instance);
    }

    /// Make sure the kernel layout is available
    fn create_kernel_layout(&mut self, ctx: &GpuContext) {
        if self.kernel_layout.checked() {
            return;
        }

        self.kernel_layout = Use::Active(KernelLayout::new(ctx, Some("Image Kernel Layout")));
    }

    /// Make sure the blur filter kernel is available
    fn create_kernel_bind(
        &mut self,
        data: &mut ImageDataInstruments,
        ctx: &GpuContext,
        params: &DrawParameters,
    ) {
        if data.kernel_bind.checked() {
            return;
        }

        data.kernel_bind = match params.raw_blur_kernel() {
            Some(kernel) => {
                self.create_kernel_layout(ctx);
                let layout = self.kernel_layout.active();
                Use::Active(KernelBinding::new(
                    ctx,
                    layout,
                    &kernel,
                    Some("Image Kernel Bind"),
                ))
            }
            None => Use::Invalid,
        };
    }

    /// Make sure the storage layout is available
    fn create_storage_layout(&mut self, ctx: &GpuContext) {
        if self.storage_layout.checked() {
            return;
        }

        self.storage_layout = Use::Active(StorageSrcDstLayout::new(
            ctx,
            Some("Image Kernel Storage Texture Bind Layout"),
        ));
    }

    /// Make sure the convolution layout is available
    fn create_convolution_layout(&mut self, ctx: &GpuContext) {
        if self.convolution_layout.checked() {
            return;
        }

        self.create_storage_layout(ctx);
        self.create_kernel_layout(ctx);
        let storage = self.storage_layout.active();
        let kernel = self.kernel_layout.active();

        let convolution = ConvolutionPipelineLayout::new(
            ctx,
            storage,
            kernel,
            Some("Image Convolution Pipeline Layout"),
        );
        self.convolution_layout = Use::Active(convolution);
    }

    /// Make sure the convolution pipeline is available
    fn create_convolution_pipeline(&mut self, ctx: &GpuContext) {
        if self.convolution_pipeline.checked() {
            return;
        }
        self.create_convolution_layout(ctx);
        let convolution = self.convolution_layout.active();

        let convolution = ConvolutionPipeline::new(
            ctx,
            convolution,
            Some("Image Convolution Shader"),
            Some("Image Convolution Pipeline"),
        );
        self.convolution_pipeline = Use::Active(convolution);
    }

    /// Make sure the storage texture copy machine is available
    fn create_copy_machine(
        &mut self,
        data: &mut ImageDataInstruments,
        ctx: &GpuContext,
        image: &Image,
    ) {
        if self.copy_machine.checked() {
            return;
        }

        self.create_original(data, ctx, image);
        let format = data.original.active().texture().format();

        let copy_machine = StorageTextureCopyMachine::new(ctx, format);
        self.copy_machine = Use::Active(copy_machine);
    }

    /// Make sure the storage data texture is available
    fn create_storage_data(
        &mut self,
        data: &mut ImageDataInstruments,
        ctx: &GpuContext,
        image: &Image,
    ) {
        if data.storage_data.checked() {
            return;
        }

        self.create_original(data, ctx, image);
        let original = data.original.active();

        let storage = SimpleStorageTexture::empty(
            ctx,
            original.texture(),
            Some("Image Data Storage Texture"),
        );
        data.storage_data = Use::Active(storage);
    }

    /// Make sure the storage scratch texture is available
    fn create_storage_scratch(
        &mut self,
        data: &mut ImageDataInstruments,
        ctx: &GpuContext,
        image: &Image,
    ) {
        if data.storage_scratch.checked() {
            return;
        }

        self.create_original(data, ctx, image);
        let original = data.original.active();

        let storage = SimpleStorageTexture::empty(
            ctx,
            original.texture(),
            Some("Image Scratch Storage Texture"),
        );
        data.storage_scratch = Use::Active(storage);
    }

    /// Make sure the lanczos metadata buffer is available
    fn create_lanczos_buffer(&mut self, ctx: &GpuContext, params: &DrawParameters) {
        match self.lanczos_buffer.take() {
            Use::Missing => {
                self.create_buffer_layout_fragment(ctx);
                let layout = self.buffer_layout_fragment.active();
                let buffer = params.raw_lanczos();
                let buffer = SimpleBuffer::new(ctx, buffer, Some("Image Lanczos Buffer"));
                let buffer =
                    SimpleBufferBind::new(ctx, buffer, layout, Some("Image Lanczos Buffer Bind"));
                self.lanczos_buffer = Use::Active(buffer);
            }
            Use::Recycle(b) | Use::Unused(b) => {
                let data = params.raw_lanczos();
                b.buffer().update(ctx, data);
                self.lanczos_buffer = Use::Active(b);
            }
            out @ (Use::Invalid | Use::Active(_)) => {
                self.lanczos_buffer = out;
            }
        }
    }

    /// Make sure the lanczos pipeline layout is available
    fn create_lanczos_pipeline_layout(&mut self, ctx: &GpuContext) {
        if self.lanczos_pipeline_layout.checked() {
            return;
        }

        self.create_texture_layout(ctx);
        self.create_buffer_layout_vertex(ctx);
        self.create_buffer_layout_fragment(ctx);
        let texture = self.texture_layout.active();
        let buffer_vertex = self.buffer_layout_vertex.active();
        let buffer_fragment = self.buffer_layout_fragment.active();

        let pipeline =
            LanczosImageRenderPipelineLayout::new(ctx, texture, buffer_vertex, buffer_fragment);
        self.lanczos_pipeline_layout = Use::Active(pipeline);
    }

    /// Make sure the lanczos pipeline is available
    fn create_lanczos_pipeline(&mut self, ctx: &GpuContext, target: &TargetContext) {
        if self.lanczos_pipeline.checked() {
            return;
        }

        self.create_lanczos_pipeline_layout(ctx);
        let layout = self.lanczos_pipeline_layout.active();

        let pipeline = RenderLanczosPipeline::new(ctx, layout, target.config.format);
        self.lanczos_pipeline = Use::Active(pipeline);
    }

    /// Make sure the blurred image is available
    fn create_blurred(
        &mut self,
        data: &mut ImageDataInstruments,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        image: &Image,
        params: &DrawParameters,
    ) {
        self.create_kernel_bind(data, ctx, params);

        if !matches!(&data.kernel_bind, Use::Active(_)) {
            data.blurred = data.blurred.take().make_unused();
            return;
        }

        let blurred = match data.blurred.take() {
            checked @ (Use::Active(_) | Use::Invalid) => {
                data.blurred = checked;
                return;
            }
            Use::Recycle(t) | Use::Unused(t) if t.texture().size() == image.extent() => t,
            Use::Recycle(_) | Use::Unused(_) | Use::Missing => {
                self.create_texture_layout(ctx);
                self.create_original(data, ctx, image);
                let texture = self.texture_layout.active();
                let original = data.original.active();

                SimpleTexture::empty(
                    ctx,
                    texture,
                    original.texture(),
                    Some("Image Blurred Texture"),
                    Some("Image Blurred Texture Bind"),
                )
            }
        };

        self.create_convolution_pipeline(ctx);
        self.create_copy_machine(data, ctx, image);
        self.create_storage_layout(ctx);
        self.create_storage_data(data, ctx, image);
        self.create_storage_scratch(data, ctx, image);
        self.create_original(data, ctx, image);

        let convolution_pipeline = self.convolution_pipeline.active();
        let copy_machine = self.copy_machine.active();
        let storage_layout = self.storage_layout.active();
        let storage_data = data.storage_data.active();
        let storage_scratch = data.storage_scratch.active();
        let original = data.original.active();

        let kernel_bind = data.kernel_bind.active();

        storage_data.copy_from_texture(
            ctx,
            encoder,
            copy_machine,
            original.texture(),
            Range::from(0..1),
        );

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Blur Compute Pass"),
                timestamp_writes: None,
            });
            convolution_pipeline.run(
                ctx,
                &mut pass,
                storage_layout,
                storage_data,
                storage_scratch,
                storage_data,
                kernel_bind,
                0,
            );
        }

        storage_data.copy_to_texture(
            ctx,
            encoder,
            copy_machine,
            blurred.texture(),
            Range::from(0..1),
        );

        data.blurred = Use::Active(blurred);
    }
}

/// Results and state changes
impl ImageMetaInstruments {
    #[expect(dead_code)]
    /// Move out all values out this object
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    /// Degrade necessary instruments to handle a replaced image
    pub fn replaced_image(&mut self) {
        self.final_output.degrade();
    }

    /// Degrade necessary instruments to handle a resized viewport
    pub fn resized(&mut self) {
        self.final_output.degrade();
        self.viewport_buffer.degrade();
    }

    /// Degrade necessary instruments to handle a new zoom level
    pub fn zoomed(&mut self) {
        self.final_output.degrade();
        self.viewport_buffer.degrade();
    }

    /// Degrade necessary instruments to handle a panned image
    pub fn panned(&mut self) {
        self.final_output.degrade();
        self.viewport_buffer.degrade();
    }

    /// Degrade necessary instruments to handle a new applied filter
    pub fn cycled_filter(&mut self) {
        self.final_output.degrade();

        self.convolution_pipeline.discard();
        self.copy_machine.discard();

        self.nearest_pipeline.discard();
        self.bilinear_pipeline.discard();
        self.lanczos_pipeline.discard();

        self.lanczos_buffer.discard();
    }
}

/// The rendering instruments for a concrete image
#[derive(Default)]
pub struct ImageDataInstruments {
    /// The current filtered result of this image
    output: Use<PassThruTexture>,
    /// The original image as a texture
    original: Use<SimpleTexture>,
    /// Quad instance
    instance: Use<InstanceBuffer<InstanceRaw>>,

    /// Storage data texture that fits the image
    storage_data: Use<SimpleStorageTexture>,
    /// Storage scratch texture that fits the image
    storage_scratch: Use<SimpleStorageTexture>,
    /// The blur kernel binding for this texture
    kernel_bind: Use<KernelBinding>,
    /// The blurred image
    blurred: Use<SimpleTexture>,
}

impl ImageDataInstruments {
    /// Create empty instruments
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the current filtered rendering result
    pub const fn output(&self) -> Option<&PassThruTexture> {
        let Use::Active(output) = &self.output else {
            return None;
        };
        Some(output)
    }

    /// Degrade necessary instruments to handle a replaced image
    pub fn replaced_image(&mut self) {
        self.original.degrade();
        self.output.degrade();
        self.blurred.degrade();
        self.storage_data.degrade();
    }

    /// Degrade necessary instruments to handle a resized viewport
    pub fn resized(&mut self) {
        self.output.discard();
        self.instance.discard();
        self.storage_data.discard();
        self.storage_scratch.discard();
        self.blurred.discard();
    }

    /// Degrade necessary instruments to handle a new zoom level
    pub fn zoomed(&mut self) {
        self.output.degrade();
        self.blurred.discard();
        self.kernel_bind.discard();
    }

    /// Degrade necessary instruments to handle a panned image
    pub fn panned(&mut self) {
        self.output.degrade();
    }

    /// Degrade necessary instruments to handle a new applied filter
    pub fn cycled_filter(&mut self) {
        self.output.degrade();

        self.storage_data.discard();
        self.storage_scratch.discard();
        self.kernel_bind.discard();
        self.blurred.discard();
    }

    /// Extract or create the render target for this image
    fn create_output(
        &mut self,
        ctx: &GpuContext,
        viewport: &ViewportGui,
        passthru: &PassThruPipeline,
    ) -> Option<PassThruTexture> {
        let Some(extent) = viewport.extent() else {
            self.output = self.output.take().make_unused();
            return None;
        };

        match self.output.take() {
            checked @ (Use::Active(_) | Use::Invalid) => {
                self.output = checked;
                None
            }
            Use::Recycle(output) | Use::Unused(output) if output.texture().size() == extent => {
                Some(output)
            }
            Use::Recycle(_) | Use::Unused(_) | Use::Missing => {
                Some(passthru.create_texture(ctx, extent))
            }
        }
    }
}
