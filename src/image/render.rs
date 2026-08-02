use std::range::Range;

use iced::wgpu;
use tristate::TriStore;

use crate::image::{DrawParameters, ImageLoaded};
use crate::instruments::bind::image::{ImageMetadataRaw, LanczosInfoRaw};
use crate::instruments::bind::storage::{SimpleStorageTexture, StorageTextureCopyMachine};
use crate::instruments::bind::texture::{SimpleTexture, SimpleTextureLayout};
use crate::instruments::buffer::{SimpleBuffer, SimpleBufferBind, SimpleBufferBindLayout};
use crate::instruments::pipeline::filter::{
    ConvolutionPipeline,
    ConvolutionPipelineLayout,
    KernelBinding,
    KernelLayout,
    StorageSrcDstLayout,
};
use crate::instruments::pipeline::image::{
    LanczosImageRenderPipelineLayout,
    RenderBilinearPipeline,
    RenderLanczosPipeline,
    RenderNearestPipeline,
    SimpleImageRenderPipelineLayout,
};
use crate::instruments::pipeline::passthru::PassThruTexture;
use crate::instruments::viewport::Viewport;
use crate::instruments::{GpuContext, TargetContext};

pub mod mipmap;

pub fn nearest(
    image: &ImageLoaded,
    instruments: &mut ImageInstruments,
    ctx: &GpuContext,
    target: &TargetContext,
    encoder: &mut wgpu::CommandEncoder,
    viewport: &Viewport,
    params: &DrawParameters,
) {
    let output = instruments
        .output
        .any_or_set_bad(|| viewport.create_texture(ctx).expect("should work"));
    if output.good().is_some() {
        return;
    }

    let texture_layout = instruments
        .texture_layout
        .good_or_set(|| SimpleTextureLayout::new(ctx, None));
    let original = instruments.original.good_or_set(|| {
        let texture = image.upload(ctx, None);
        SimpleTexture::new(ctx, texture_layout, texture, None)
    });

    let buffer_layout = instruments
        .buffer_layout
        .good_or_set(|| SimpleBufferBindLayout::new(ctx, None));
    let pipeline_layout = instruments
        .simple_pipeline_layout
        .good_or_set(|| SimpleImageRenderPipelineLayout::new(ctx, texture_layout, buffer_layout));
    let pipeline = instruments
        .nearest_pipeline
        .good_or_set(|| RenderNearestPipeline::new(ctx, pipeline_layout, target.config.format));

    let meta = params.raw_metadata();
    let meta_buffer = instruments
        .meta_buffer
        .any_or_set_good(|| {
            let meta_buffer = SimpleBuffer::new(ctx, meta, None);
            SimpleBufferBind::new(ctx, meta_buffer, buffer_layout, None)
        })
        .or_update(|b| b.buffer().update(ctx, meta));

    output.or_update(|output| {
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
    });
}

pub fn bilinear(
    image: &ImageLoaded,
    instruments: &mut ImageInstruments,
    ctx: &GpuContext,
    target: &TargetContext,
    encoder: &mut wgpu::CommandEncoder,
    viewport: &Viewport,
    params: &DrawParameters,
) {
    let output = instruments
        .output
        .any_or_set_bad(|| viewport.create_texture(ctx).expect("should work"));
    if output.good().is_some() {
        return;
    }

    let texture_layout = instruments
        .texture_layout
        .good_or_set(|| SimpleTextureLayout::new(ctx, None));
    let original = instruments.original.good_or_set(|| {
        let texture = image.upload(ctx, None);
        SimpleTexture::new(ctx, texture_layout, texture, None)
    });

    let buffer_layout = instruments
        .buffer_layout
        .good_or_set(|| SimpleBufferBindLayout::new(ctx, None));
    let pipeline_layout = instruments
        .simple_pipeline_layout
        .good_or_set(|| SimpleImageRenderPipelineLayout::new(ctx, texture_layout, buffer_layout));
    let pipeline = instruments
        .bilinear_pipeline
        .good_or_set(|| RenderBilinearPipeline::new(ctx, pipeline_layout, target.config.format));

    let meta = params.raw_metadata();
    let meta_buffer = instruments
        .meta_buffer
        .any_or_set_good(|| {
            let meta_buffer = SimpleBuffer::new(ctx, meta, None);
            SimpleBufferBind::new(ctx, meta_buffer, buffer_layout, None)
        })
        .or_update(|b| b.buffer().update(ctx, meta));

    output.or_update(|output| {
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
    });
}

#[expect(clippy::too_many_lines)]
pub fn lanczos(
    image: &ImageLoaded,
    instruments: &mut ImageInstruments,
    ctx: &GpuContext,
    target: &TargetContext,
    encoder: &mut wgpu::CommandEncoder,
    viewport: &Viewport,
    params: &DrawParameters,
) {
    let output = instruments
        .output
        .any_or_set_bad(|| viewport.create_texture(ctx).expect("should work"));
    if output.good().is_some() {
        return;
    }

    let texture_layout = instruments
        .texture_layout
        .good_or_set(|| SimpleTextureLayout::new(ctx, None));
    let original = instruments.original.good_or_set(|| {
        let texture = image.upload(ctx, None);
        SimpleTexture::new(ctx, texture_layout, texture, None)
    });
    let original_texture = original.texture();

    // Blur image with gaussian filter
    let kernel_layout = instruments
        .kernel_layout
        .good_or_set(|| KernelLayout::new(ctx, None));

    let kernel_bind = instruments.kernel_bind.any_or_try_set_good(|| {
        params
            .raw_blur_kernel()
            .map(|kernel| KernelBinding::new(ctx, kernel_layout, &kernel, None))
    });

    let blurred = match kernel_bind {
        None => original,
        Some(kernel_bind) => instruments.blurred.good_or_set(|| {
            let storage_layout = instruments
                .storage_layout
                .good_or_set(|| StorageSrcDstLayout::new(ctx, None));
            let convolution_layout = instruments.convolution_layout.good_or_set(|| {
                ConvolutionPipelineLayout::new(ctx, storage_layout, kernel_layout, None)
            });
            let convolution_pipeline = instruments
                .convolution_pipeline
                .good_or_set(|| ConvolutionPipeline::new(ctx, convolution_layout, None));

            let copy_machine = instruments
                .copy_machine
                .good_or_set(|| StorageTextureCopyMachine::new(ctx, original_texture.format()));

            let storage_data = instruments
                .storage_data
                .good_or_set(|| SimpleStorageTexture::empty(ctx, original_texture, None));
            let storage_scratch = instruments
                .storage_scratch
                .good_or_set(|| SimpleStorageTexture::empty(ctx, original_texture, None));
            storage_data.copy_from_texture(
                ctx,
                encoder,
                copy_machine,
                original_texture,
                Range::from(0..1),
            );

            convolution_pipeline.run(
                ctx,
                encoder,
                storage_layout,
                storage_data,
                storage_scratch,
                kernel_bind.good().expect("no bad kernel bind ever"),
                0,
            );

            let blurred = SimpleTexture::empty(ctx, texture_layout, original_texture, None);
            storage_data.copy_to_texture(
                ctx,
                encoder,
                copy_machine,
                blurred.texture(),
                Range::from(0..1),
            );
            blurred
        }),
    };

    // Interpolate with Lanczos filter
    let buffer_layout = instruments
        .buffer_layout
        .good_or_set(|| SimpleBufferBindLayout::new(ctx, None));
    let pipeline_layout = instruments
        .lanczos_pipeline_layout
        .good_or_set(|| LanczosImageRenderPipelineLayout::new(ctx, texture_layout, buffer_layout));
    let pipeline = instruments
        .lanczos_pipeline
        .good_or_set(|| RenderLanczosPipeline::new(ctx, pipeline_layout, target.config.format));

    let meta = params.raw_metadata();
    let meta_buffer = instruments
        .meta_buffer
        .any_or_set_good(|| {
            let meta_buffer = SimpleBuffer::new(ctx, meta, None);
            SimpleBufferBind::new(ctx, meta_buffer, buffer_layout, None)
        })
        .or_update(|b| b.buffer().update(ctx, meta));
    let lanczos = params.raw_lanczos();
    let lanczos_buffer = instruments
        .lanczos_buffer
        .any_or_set_good(|| {
            let lanczos = SimpleBuffer::new(ctx, lanczos, None);
            SimpleBufferBind::new(ctx, lanczos, buffer_layout, None)
        })
        .or_update(|b| b.buffer().update(ctx, lanczos));

    output.or_update(|output| {
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
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFilter {
    #[default]
    Nearest,
    BiLinear,
    Lanczos,
}

#[derive(Default)]
pub struct ImageInstruments {
    output: TriStore<PassThruTexture>,
    original: TriStore<SimpleTexture>,
    params: TriStore<DrawParameters>,

    storage_layout: TriStore<StorageSrcDstLayout>,
    kernel_layout: TriStore<KernelLayout>,
    convolution_layout: TriStore<ConvolutionPipelineLayout>,
    convolution_pipeline: TriStore<ConvolutionPipeline>,

    copy_machine: TriStore<StorageTextureCopyMachine>,
    storage_data: TriStore<SimpleStorageTexture>,
    storage_scratch: TriStore<SimpleStorageTexture>,
    kernel_bind: TriStore<KernelBinding>,
    blurred: TriStore<SimpleTexture>,

    texture_layout: TriStore<SimpleTextureLayout>,
    buffer_layout: TriStore<SimpleBufferBindLayout>,

    simple_pipeline_layout: TriStore<SimpleImageRenderPipelineLayout>,
    lanczos_pipeline_layout: TriStore<LanczosImageRenderPipelineLayout>,
    nearest_pipeline: TriStore<RenderNearestPipeline>,
    bilinear_pipeline: TriStore<RenderBilinearPipeline>,
    lanczos_pipeline: TriStore<RenderLanczosPipeline>,

    meta_buffer: TriStore<SimpleBufferBind<ImageMetadataRaw>>,
    lanczos_buffer: TriStore<SimpleBufferBind<LanczosInfoRaw>>,
}

impl ImageInstruments {
    pub const fn output(&self) -> &TriStore<PassThruTexture> {
        &self.output
    }

    #[expect(dead_code)]
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
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
        self.meta_buffer.degrade();
    }

    pub fn panned(&mut self) {
        self.output.degrade();
        self.meta_buffer.degrade();
    }

    pub fn cycled_filter(&mut self) {
        self.output.degrade();
        self.original.keep();
        self.params.keep();

        self.storage_layout.discard();
        self.kernel_layout.discard();
        self.convolution_layout.discard();
        self.convolution_pipeline.discard();

        self.copy_machine.discard();
        self.storage_data.discard();
        self.storage_scratch.discard();
        self.kernel_bind.discard();
        self.blurred.discard();

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
