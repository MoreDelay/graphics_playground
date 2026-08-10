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
use crate::instruments::{GpuContext, TargetContext};

#[derive(Default)]
pub struct ImageMetaInstruments {
    final_output: Use<PassThruTexture>,

    storage_layout: Use<StorageSrcDstLayout>,
    kernel_layout: Use<KernelLayout>,
    convolution_layout: Use<ConvolutionPipelineLayout>,
    convolution_pipeline: Use<ConvolutionPipeline>,
    copy_machine: Use<StorageTextureCopyMachine>,

    texture_layout: Use<SimpleTextureLayout>,
    buffer_layout: Use<SimpleBufferBindLayout>,

    simple_pipeline_layout: Use<SimpleImageRenderPipelineLayout>,
    lanczos_pipeline_layout: Use<LanczosImageRenderPipelineLayout>,
    nearest_pipeline: Use<RenderNearestPipeline>,
    bilinear_pipeline: Use<RenderBilinearPipeline>,
    lanczos_pipeline: Use<RenderLanczosPipeline>,

    meta_buffer: Use<SimpleBufferBind<ImageMetadataRaw>>,
    lanczos_buffer: Use<SimpleBufferBind<LanczosInfoRaw>>,
}

/// Public API
impl ImageMetaInstruments {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn final_output(&self) -> Option<&PassThruTexture> {
        let Use::Active(output) = &self.final_output else {
            return None;
        };
        Some(output)
    }

    pub fn set_final_output(&mut self, output: PassThruTexture) -> &PassThruTexture {
        self.final_output = Use::Active(output);
        self.final_output.active()
    }

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

    fn create_texture_layout(&mut self, ctx: &GpuContext) {
        if self.texture_layout.checked() {
            return;
        }

        let out = SimpleTextureLayout::new(ctx, Some("Image Texture Layout"));
        self.texture_layout = Use::Active(out);
    }

    fn create_buffer_layout(&mut self, ctx: &GpuContext) {
        if self.buffer_layout.checked() {
            return;
        }
        let out = SimpleBufferBindLayout::new(ctx, Some("Image Buffer Layout"));
        self.buffer_layout = Use::Active(out);
    }

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

    fn create_nearest_pipeline(&mut self, ctx: &GpuContext, target: &TargetContext) {
        if self.nearest_pipeline.checked() {
            return;
        }

        self.create_simple_pipeline_layout(ctx);
        let pipeline = self.simple_pipeline_layout.active();

        let pipeline = RenderNearestPipeline::new(ctx, pipeline, target.config.format);
        self.nearest_pipeline = Use::Active(pipeline);
    }

    fn create_bilinear_pipeline(&mut self, ctx: &GpuContext, target: &TargetContext) {
        if self.bilinear_pipeline.checked() {
            return;
        }

        self.create_simple_pipeline_layout(ctx);
        let pipeline = self.simple_pipeline_layout.active();

        let pipeline = RenderBilinearPipeline::new(ctx, pipeline, target.config.format);
        self.bilinear_pipeline = Use::Active(pipeline);
    }

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

    fn create_kernel_layout(&mut self, ctx: &GpuContext) {
        if self.kernel_layout.checked() {
            return;
        }

        self.kernel_layout = Use::Active(KernelLayout::new(ctx, Some("Image Kernel Layout")));
    }

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

    fn create_storage_layout(&mut self, ctx: &GpuContext) {
        if self.storage_layout.checked() {
            return;
        }

        self.storage_layout = Use::Active(StorageSrcDstLayout::new(
            ctx,
            Some("Image Kernel Storage Texture Bind Layout"),
        ));
    }

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

    fn create_lanczos_pipeline(&mut self, ctx: &GpuContext, target: &TargetContext) {
        if self.lanczos_pipeline.checked() {
            return;
        }

        self.create_lanczos_pipeline_layout(ctx);
        let layout = self.lanczos_pipeline_layout.active();

        let pipeline = RenderLanczosPipeline::new(ctx, layout, target.config.format);
        self.lanczos_pipeline = Use::Active(pipeline);
    }

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
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    pub fn uncheck_all(&mut self) {
        self.final_output.uncheck();

        self.storage_layout.uncheck();
        self.kernel_layout.uncheck();
        self.convolution_layout.uncheck();
        self.convolution_pipeline.uncheck();
        self.copy_machine.uncheck();

        self.texture_layout.uncheck();
        self.buffer_layout.uncheck();

        self.simple_pipeline_layout.uncheck();
        self.lanczos_pipeline_layout.uncheck();
        self.nearest_pipeline.uncheck();
        self.bilinear_pipeline.uncheck();
        self.lanczos_pipeline.uncheck();

        self.meta_buffer.uncheck();
        self.lanczos_buffer.uncheck();
    }

    pub fn replaced_image(&mut self) {
        self.final_output.degrade();
        self.meta_buffer.degrade();
    }

    pub fn resized(&mut self) {
        self.final_output.degrade();
    }

    pub fn zoomed(&mut self) {
        self.final_output.degrade();
        self.meta_buffer.degrade();
    }

    pub fn panned(&mut self) {
        self.final_output.degrade();
        self.meta_buffer.degrade();
    }

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

        self.meta_buffer.keep();
        self.lanczos_buffer.discard();
    }
}

#[derive(Default)]
pub struct ImageDataInstruments {
    output: Use<PassThruTexture>,
    original: Use<SimpleTexture>,
    params: Use<DrawParameters>,

    storage_data: Use<SimpleStorageTexture>,
    storage_scratch: Use<SimpleStorageTexture>,
    kernel_bind: Use<KernelBinding>,
    blurred: Use<SimpleTexture>,
}

impl ImageDataInstruments {
    pub fn new() -> Self {
        Self::default()
    }

    pub const fn output(&self) -> Option<&PassThruTexture> {
        let Use::Active(output) = &self.output else {
            return None;
        };
        Some(output)
    }

    pub fn uncheck_all(&mut self) {
        self.output.uncheck();
        self.original.uncheck();
        self.params.uncheck();

        self.storage_data.uncheck();
        self.storage_scratch.uncheck();
        self.kernel_bind.uncheck();
        self.blurred.uncheck();
    }

    pub fn replaced_image(&mut self) {
        self.original.degrade();
        self.output.degrade();
        self.blurred.degrade();
        self.storage_data.degrade();
    }

    pub fn resized(&mut self) {
        self.output.discard();
        self.storage_data.discard();
        self.storage_scratch.discard();
        self.blurred.discard();
    }

    pub fn zoomed(&mut self) {
        self.output.degrade();
        self.blurred.discard();
        self.kernel_bind.discard();
    }

    pub fn panned(&mut self) {
        self.output.degrade();
    }

    pub fn cycled_filter(&mut self) {
        self.output.degrade();
        self.original.keep();
        self.params.keep();

        self.storage_data.discard();
        self.storage_scratch.discard();
        self.kernel_bind.discard();
        self.blurred.discard();
    }

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

#[derive(Default)]
enum Use<T> {
    #[default]
    Missing,
    Invalid,
    Active(T),
    Recycle(T),
    Unused(T),
}

impl<T> Use<T> {
    fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    const fn checked(&self) -> bool {
        matches!(self, Self::Active(_) | Self::Unused(_))
    }

    fn uncheck(&mut self) {
        *self = match self.take() {
            Self::Missing | Self::Invalid => Self::Missing,
            Self::Active(v) => Self::Active(v),
            Self::Recycle(v) | Self::Unused(v) => Self::Recycle(v),
        }
    }

    fn active(&self) -> &T {
        let Self::Active(v) = self else {
            panic!("value is not in active use");
        };
        v
    }

    fn maybe_active(&self) -> Option<&T> {
        match self {
            Self::Missing | Self::Recycle(_) => panic!("value still unchecked"),
            Self::Active(v) => Some(v),
            Self::Invalid | Self::Unused(_) => None,
        }
    }

    fn degrade(&mut self) {
        *self = match self.take() {
            Self::Missing => Self::Invalid,
            Self::Invalid => Self::Invalid,
            Self::Active(v) => Self::Unused(v),
            Self::Recycle(v) => Self::Unused(v),
            Self::Unused(v) => Self::Unused(v),
        }
    }

    fn discard(&mut self) {
        *self = Self::Missing;
    }

    #[expect(clippy::unused_self)]
    const fn keep(&self) {}

    fn make_unused(self) -> Self {
        match self {
            Self::Missing | Self::Invalid => Self::Invalid,
            Self::Active(v) | Self::Recycle(v) | Self::Unused(v) => Self::Unused(v),
        }
    }
}
