//! Orchestration to render the image viewer widget

use std::range::Range;

use iced::wgpu;

use crate::image::{DrawParameters, ImageMemory};
use crate::instruments::bind::image::{ImageMetadataRaw, LanczosInfoRaw};
use crate::instruments::bind::storage::{
    SimpleStorageTexture,
    StorageSrcDstLayout,
    StorageTextureCopyMachine,
};
use crate::instruments::bind::texture::{SimpleTexture, SimpleTextureLayout};
use crate::instruments::buffer::{SimpleBuffer, SimpleBufferBind, SimpleBufferBindLayout};
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
use crate::instruments::viewport::Viewport;
use crate::instruments::{GpuContext, TargetContext, Use};

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
    /// The universal buffer bind group layout
    buffer_layout: Use<SimpleBufferBindLayout>,

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

    /// The buffer for image metadata used in all render pipelines
    meta_buffer: Use<SimpleBufferBind<ImageMetadataRaw>>,
    /// The buffer for lanczos filter metadata
    lanczos_buffer: Use<SimpleBufferBind<LanczosInfoRaw>>,
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
    #[expect(clippy::too_many_arguments)]
    pub fn nearest(
        &mut self,
        data: &mut ImageDataInstruments,
        ctx: &GpuContext,
        target: &TargetContext,
        passthru: &PassThruPipeline,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        image: &ImageMemory,
        params: &DrawParameters,
    ) {
        let Some(output) = data.create_output(ctx, viewport, passthru) else {
            return;
        };

        self.create_original(data, ctx, image);
        self.create_meta_buffer(ctx, params);
        self.create_nearest_pipeline(ctx, target);

        let original = data.original.active();
        let meta_buffer = self.meta_buffer.active();
        let pipeline = self.nearest_pipeline.active();

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

        pipeline.draw(&mut pass, original, meta_buffer);

        data.output = Use::Active(output);
    }

    /// Render a new image with the provided [`ImageDataInstruments`] using "bilinear" filter
    #[expect(clippy::too_many_arguments)]
    pub fn bilinear(
        &mut self,
        data: &mut ImageDataInstruments,
        ctx: &GpuContext,
        target: &TargetContext,
        passthru: &PassThruPipeline,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        image: &ImageMemory,
        params: &DrawParameters,
    ) {
        let Some(output) = data.create_output(ctx, viewport, passthru) else {
            return;
        };

        self.create_meta_buffer(ctx, params);
        self.create_original(data, ctx, image);
        self.create_bilinear_pipeline(ctx, target);

        let meta_buffer = self.meta_buffer.active();
        let original = data.original.active();
        let pipeline = self.bilinear_pipeline.active();

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

        pipeline.draw(&mut pass, original, meta_buffer);

        data.output = Use::Active(output);
    }

    /// Render a new image with the provided [`ImageDataInstruments`] using "lanczos" filter
    #[expect(clippy::too_many_arguments)]
    pub fn lanczos(
        &mut self,
        data: &mut ImageDataInstruments,
        ctx: &GpuContext,
        target: &TargetContext,
        passthru: &PassThruPipeline,
        encoder: &mut wgpu::CommandEncoder,
        viewport: &Viewport,
        image: &ImageMemory,
        params: &DrawParameters,
    ) {
        let Some(output) = data.create_output(ctx, viewport, passthru) else {
            return;
        };

        // Interpolate with Lanczos filter
        self.create_blurred(data, ctx, encoder, image, params);
        self.create_lanczos_pipeline(ctx, target);
        self.create_meta_buffer(ctx, params);
        self.create_lanczos_buffer(ctx, params);

        let blurred = if let Some(blurred) = &data.blurred.maybe_active() {
            blurred
        } else {
            self.create_original(data, ctx, image);
            data.original.active()
        };

        let pipeline = self.lanczos_pipeline.active();
        let meta_buffer = self.meta_buffer.active();
        let lanczos_buffer = self.lanczos_buffer.active();

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

        pipeline.draw(&mut pass, blurred, meta_buffer, lanczos_buffer);

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
        viewport: &Viewport,
    ) -> Option<PassThruTexture> {
        let Some(extent) = viewport.extent() else {
            self.final_output = self.final_output.take().make_unused();
            return None;
        };

        match self.final_output.take() {
            checked @ (Use::Active(_) | Use::Invalid) => {
                self.final_output = checked;
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

    /// Make sure the texture layout is available
    fn create_texture_layout(&mut self, ctx: &GpuContext) {
        if self.texture_layout.checked() {
            return;
        }

        let out = SimpleTextureLayout::new(ctx, Some("Image Texture Layout"));
        self.texture_layout = Use::Active(out);
    }

    /// Make sure the buffer layout is available
    fn create_buffer_layout(&mut self, ctx: &GpuContext) {
        if self.buffer_layout.checked() {
            return;
        }
        let out = SimpleBufferBindLayout::new(ctx, Some("Image Buffer Layout"));
        self.buffer_layout = Use::Active(out);
    }

    /// Make sure the simple pipeline layout is available
    fn create_simple_pipeline_layout(&mut self, ctx: &GpuContext) {
        if self.simple_pipeline_layout.checked() {
            return;
        }

        self.create_texture_layout(ctx);
        self.create_buffer_layout(ctx);

        let texture = self.texture_layout.active();
        let buffer = self.buffer_layout.active();

        let pipeline = SimpleImageRenderPipelineLayout::new(ctx, texture, buffer);
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
    fn create_meta_buffer(&mut self, ctx: &GpuContext, params: &DrawParameters) {
        match self.meta_buffer.take() {
            Use::Missing => {
                self.create_buffer_layout(ctx);
                let buffer_layout = self.buffer_layout.active();

                let meta = params.raw_metadata();
                let meta_buffer = SimpleBuffer::new(ctx, meta, Some("Image Metainfo Buffer"));
                let meta_buffer = SimpleBufferBind::new(
                    ctx,
                    meta_buffer,
                    buffer_layout,
                    Some("Image Metainfo Binding"),
                );
                self.meta_buffer = Use::Active(meta_buffer);
            }
            Use::Recycle(b) | Use::Unused(b) => {
                let meta = params.raw_metadata();
                b.buffer().update(ctx, meta);
                self.meta_buffer = Use::Active(b);
            }
            out @ (Use::Invalid | Use::Active(_)) => {
                self.meta_buffer = out;
            }
        }
    }

    /// Make sure the original image has been uploaded to a texture
    fn create_original(
        &mut self,
        data: &mut ImageDataInstruments,
        ctx: &GpuContext,
        image: &ImageMemory,
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
        image: &ImageMemory,
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
        image: &ImageMemory,
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
        image: &ImageMemory,
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
                self.create_buffer_layout(ctx);
                let layout = self.buffer_layout.active();
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
        self.create_buffer_layout(ctx);
        let texture = self.texture_layout.active();
        let buffer = self.buffer_layout.active();

        let pipeline = LanczosImageRenderPipelineLayout::new(ctx, texture, buffer);
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
        image: &ImageMemory,
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
        self.meta_buffer.degrade();
    }

    /// Degrade necessary instruments to handle a resized viewport
    pub fn resized(&mut self) {
        self.final_output.degrade();
    }

    /// Degrade necessary instruments to handle a new zoom level
    pub fn zoomed(&mut self) {
        self.final_output.degrade();
        self.meta_buffer.degrade();
    }

    /// Degrade necessary instruments to handle a panned image
    pub fn panned(&mut self) {
        self.final_output.degrade();
        self.meta_buffer.degrade();
    }

    /// Degrade necessary instruments to handle a new applied filter
    pub fn cycled_filter(&mut self) {
        self.final_output.degrade();

        self.storage_layout.discard();
        self.kernel_layout.discard();
        self.convolution_layout.discard();
        self.convolution_pipeline.discard();
        self.copy_machine.discard();

        self.texture_layout.discard();
        self.buffer_layout.discard();

        self.simple_pipeline_layout.discard();
        self.lanczos_pipeline_layout.discard();
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
        viewport: &Viewport,
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
