use std::range::Range;

use iced::wgpu;

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
        .or_set_outdated(|| viewport.create_texture(ctx).expect("should work"));
    if let InstrumentAvailable::Valid(_) = output {
        return;
    }

    let texture_layout = instruments
        .texture_layout
        .or_replace(|| SimpleTextureLayout::new(ctx, None));
    let original = instruments.original.or_replace(|| {
        let texture = image.upload(ctx, None);
        SimpleTexture::new(ctx, texture_layout, texture, None)
    });

    let buffer_layout = instruments
        .buffer_layout
        .or_replace(|| SimpleBufferBindLayout::new(ctx, None));
    let pipeline_layout = instruments
        .simple_pipeline_layout
        .or_replace(|| SimpleImageRenderPipelineLayout::new(ctx, texture_layout, buffer_layout));
    let pipeline = instruments
        .nearest_pipeline
        .or_replace(|| RenderNearestPipeline::new(ctx, pipeline_layout, target.config.format));

    let meta = params.raw_metadata();
    let meta_buffer = instruments
        .meta_buffer
        .or_set(|| {
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
        .or_set_outdated(|| viewport.create_texture(ctx).expect("should work"));
    if let InstrumentAvailable::Valid(_) = output {
        return;
    }

    let texture_layout = instruments
        .texture_layout
        .or_replace(|| SimpleTextureLayout::new(ctx, None));
    let original = instruments.original.or_replace(|| {
        let texture = image.upload(ctx, None);
        SimpleTexture::new(ctx, texture_layout, texture, None)
    });

    let buffer_layout = instruments
        .buffer_layout
        .or_replace(|| SimpleBufferBindLayout::new(ctx, None));
    let pipeline_layout = instruments
        .simple_pipeline_layout
        .or_replace(|| SimpleImageRenderPipelineLayout::new(ctx, texture_layout, buffer_layout));
    let pipeline = instruments
        .bilinear_pipeline
        .or_replace(|| RenderBilinearPipeline::new(ctx, pipeline_layout, target.config.format));

    let meta = params.raw_metadata();
    let meta_buffer = instruments
        .meta_buffer
        .or_set(|| {
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
    println!("run lanczos");

    let output = instruments
        .output
        .or_set_outdated(|| viewport.create_texture(ctx).expect("should work"));
    if let InstrumentAvailable::Valid(_) = output {
        return;
    }

    let texture_layout = instruments
        .texture_layout
        .or_replace(|| SimpleTextureLayout::new(ctx, None));
    let original = instruments.original.or_replace(|| {
        let texture = image.upload(ctx, None);
        SimpleTexture::new(ctx, texture_layout, texture, None)
    });
    let original_texture = original.texture();

    let (storage_layout, kernel_layout, convolution_pipeline) = {
        let storage = instruments
            .storage_layout
            .or_replace(|| StorageSrcDstLayout::new(ctx, None));
        let kernel = instruments
            .kernel_layout
            .or_replace(|| KernelLayout::new(ctx, None));
        let convolution = instruments
            .convolution_layout
            .or_replace(|| ConvolutionPipelineLayout::new(ctx, storage, kernel, None));
        let pipeline = instruments
            .convolution_pipeline
            .or_replace(|| ConvolutionPipeline::new(ctx, convolution, None));
        (storage, kernel, pipeline)
    };

    let copy_machine = instruments
        .copy_machine
        .or_replace(|| StorageTextureCopyMachine::new(ctx, original_texture.format()));

    let (storage_data, storage_scratch) = {
        let storage_data = instruments
            .storage_data
            .or_replace(|| SimpleStorageTexture::empty(ctx, original_texture, None));
        let storage_scratch = instruments
            .storage_scratch
            .or_replace(|| SimpleStorageTexture::empty(ctx, original_texture, None));
        (storage_data, storage_scratch)
    };
    storage_data.copy_from_texture(
        ctx,
        encoder,
        copy_machine,
        original_texture,
        Range::from(0..1),
    );

    let kernel_bind = instruments.kernel_bind.or_replace(|| {
        let kernel = params.raw_blur_kernel();
        KernelBinding::new(ctx, kernel_layout, &kernel, None)
    });
    convolution_pipeline.run(
        ctx,
        encoder,
        storage_layout,
        storage_data,
        storage_scratch,
        kernel_bind,
        0,
    );

    let blurred = instruments
        .blurred
        .or_replace(|| SimpleTexture::empty(ctx, texture_layout, original_texture, None));
    storage_data.copy_to_texture(
        ctx,
        encoder,
        copy_machine,
        blurred.texture(),
        Range::from(0..1),
    );

    let buffer_layout = instruments
        .buffer_layout
        .or_replace(|| SimpleBufferBindLayout::new(ctx, None));
    let pipeline_layout = instruments
        .lanczos_pipeline_layout
        .or_replace(|| LanczosImageRenderPipelineLayout::new(ctx, texture_layout, buffer_layout));
    let pipeline = instruments
        .lanczos_pipeline
        .or_replace(|| RenderLanczosPipeline::new(ctx, pipeline_layout, target.config.format));

    let meta = params.raw_metadata();
    let meta_buffer = instruments
        .meta_buffer
        .or_set(|| {
            let meta_buffer = SimpleBuffer::new(ctx, meta, None);
            SimpleBufferBind::new(ctx, meta_buffer, buffer_layout, None)
        })
        .or_update(|b| b.buffer().update(ctx, meta));
    let lanczos = params.raw_lanczos();
    let lanczos_buffer = instruments
        .lanczos_buffer
        .or_set(|| {
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
pub enum Instrument<T> {
    #[default]
    Missing,
    Available(InstrumentAvailable<T>),
}

impl<T> Instrument<T> {
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    #[expect(unused)]
    pub fn unwrap_outdated(self) -> T {
        match self {
            Self::Missing => panic!("value is missing"),
            Self::Available(t) => t.unwrap_outdated(),
        }
    }

    #[expect(unused)]
    pub fn unwrap(self) -> T {
        match self {
            Self::Missing => panic!("value is missing"),
            Self::Available(t) => t.unwrap(),
        }
    }

    #[expect(unused)]
    pub fn unwrap_any(self) -> T {
        match self {
            Self::Missing => panic!("value is missing"),
            Self::Available(t) => t.unwrap_any(),
        }
    }

    pub fn or_set<F>(&mut self, f: F) -> &mut InstrumentAvailable<T>
    where
        F: FnOnce() -> T,
    {
        let f = || InstrumentAvailable::Valid(f());
        self.or_use(f)
    }

    pub fn or_set_outdated<F>(&mut self, f: F) -> &mut InstrumentAvailable<T>
    where
        F: FnOnce() -> T,
    {
        match self {
            Self::Missing => {
                let inner = InstrumentAvailable::OutOfDate(f());
                *self = Self::Available(inner);
            }
            Self::Available(_) => (),
        }
        self.get_available_mut().expect("just set above")
    }

    pub fn or_use<F>(&mut self, f: F) -> &mut InstrumentAvailable<T>
    where
        F: FnOnce() -> InstrumentAvailable<T>,
    {
        match self {
            Self::Missing => {
                *self = Self::Available(f());
            }
            Self::Available(_) => (),
        }
        self.get_available_mut().expect("just set above")
    }

    #[expect(unused)]
    pub fn or_else<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce() -> Self,
    {
        match self {
            Self::Available(InstrumentAvailable::Valid(_)) => (),
            Self::Missing | Self::Available(_) => *self = f(),
        }
        self
    }

    #[expect(unused)]
    pub fn outdated_or_else<F>(&mut self, f: F) -> &mut Self
    where
        F: FnOnce() -> Self,
    {
        match self {
            Self::Available(InstrumentAvailable::OutOfDate(_)) => (),
            Self::Missing | Self::Available(_) => *self = f(),
        }
        self
    }

    #[expect(unused)]
    pub fn replace(&mut self, t: T) -> &mut T {
        *self = Self::Available(InstrumentAvailable::Valid(t));
        self.get_mut().expect("set above")
    }

    pub fn or_replace<F>(&mut self, f: F) -> &mut T
    where
        F: FnOnce() -> T,
    {
        match self {
            Self::Available(InstrumentAvailable::Valid(_)) => (),
            Self::Missing | Self::Available(_) => {
                *self = Self::Available(InstrumentAvailable::Valid(f()));
            }
        }
        self.get_mut().expect("just set above")
    }

    #[expect(clippy::unused_self)]
    pub const fn keep(&self) {}

    pub fn out_of_date(&mut self) {
        let next = match self.take() {
            Self::Missing => Self::Missing,
            Self::Available(mut t) => {
                t.out_of_date();
                Self::Available(t)
            }
        };
        *self = next;
    }

    pub fn reset(&mut self) {
        let next = match self.take() {
            Self::Missing => Self::Missing,
            Self::Available(_) => Self::Missing,
        };
        *self = next;
    }

    pub fn get(&self) -> Option<&T> {
        match self {
            Self::Missing => None,
            Self::Available(t) => t.get(),
        }
    }

    #[expect(unused)]
    pub fn get_any(&self) -> Option<&T> {
        match self {
            Self::Missing => None,
            Self::Available(t) => Some(t.get_any()),
        }
    }

    #[expect(unused)]
    pub const fn get_available(&self) -> Option<&InstrumentAvailable<T>> {
        match self {
            Self::Missing => None,
            Self::Available(t) => Some(t),
        }
    }

    pub fn get_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Missing => None,
            Self::Available(t) => t.get_mut(),
        }
    }

    #[expect(unused)]
    pub fn get_any_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Missing => None,
            Self::Available(t) => Some(t.get_any_mut()),
        }
    }

    pub const fn get_available_mut(&mut self) -> Option<&mut InstrumentAvailable<T>> {
        match self {
            Self::Missing => None,
            Self::Available(t) => Some(t),
        }
    }
}

pub enum InstrumentAvailable<T> {
    Transition,
    OutOfDate(T),
    Valid(T),
}

impl<T> InstrumentAvailable<T> {
    pub fn unwrap_outdated(self) -> T {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => t,
            Self::Valid(_) => panic!("value is valid"),
        }
    }

    pub fn unwrap(self) -> T {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(_) => panic!("value is out of date"),
            Self::Valid(t) => t,
        }
    }

    pub fn unwrap_any(self) -> T {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => t,
            Self::Valid(t) => t,
        }
    }

    pub fn or_update<F>(&mut self, f: F) -> &mut T
    where
        F: FnOnce(&mut T),
    {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => {
                f(t);
                let t = self.take().unwrap_outdated();
                *self = Self::Valid(t);
            }
            Self::Valid(_) => (),
        }
        self.get_mut().expect("just set to valid")
    }

    pub const fn take(&mut self) -> Self {
        std::mem::replace(self, Self::Transition)
    }

    pub fn out_of_date(&mut self) {
        *self = match self.take() {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => Self::OutOfDate(t),
            Self::Valid(t) => Self::OutOfDate(t),
        };
    }

    pub fn get(&self) -> Option<&T> {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(_) => None,
            Self::Valid(t) => Some(t),
        }
    }

    pub fn get_any(&self) -> &T {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => t,
            Self::Valid(t) => t,
        }
    }

    pub fn get_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(_) => None,
            Self::Valid(t) => Some(t),
        }
    }

    pub fn get_any_mut(&mut self) -> &mut T {
        match self {
            Self::Transition => unreachable!(),
            Self::OutOfDate(t) => t,
            Self::Valid(t) => t,
        }
    }
}

#[derive(Default)]
pub struct ImageInstruments {
    output: Instrument<PassThruTexture>,
    original: Instrument<SimpleTexture>,
    params: Instrument<DrawParameters>,

    storage_layout: Instrument<StorageSrcDstLayout>,
    kernel_layout: Instrument<KernelLayout>,
    convolution_layout: Instrument<ConvolutionPipelineLayout>,
    convolution_pipeline: Instrument<ConvolutionPipeline>,

    copy_machine: Instrument<StorageTextureCopyMachine>,
    storage_data: Instrument<SimpleStorageTexture>,
    storage_scratch: Instrument<SimpleStorageTexture>,
    kernel_bind: Instrument<KernelBinding>,
    blurred: Instrument<SimpleTexture>,

    texture_layout: Instrument<SimpleTextureLayout>,
    buffer_layout: Instrument<SimpleBufferBindLayout>,

    simple_pipeline_layout: Instrument<SimpleImageRenderPipelineLayout>,
    lanczos_pipeline_layout: Instrument<LanczosImageRenderPipelineLayout>,
    nearest_pipeline: Instrument<RenderNearestPipeline>,
    bilinear_pipeline: Instrument<RenderBilinearPipeline>,
    lanczos_pipeline: Instrument<RenderLanczosPipeline>,

    meta_buffer: Instrument<SimpleBufferBind<ImageMetadataRaw>>,
    lanczos_buffer: Instrument<SimpleBufferBind<LanczosInfoRaw>>,
}

impl ImageInstruments {
    pub const fn output(&self) -> &Instrument<PassThruTexture> {
        &self.output
    }

    #[expect(unused)]
    pub fn take(&mut self) -> Self {
        std::mem::take(self)
    }

    pub fn replaced_image(&mut self) {
        self.original.out_of_date();
        self.output.out_of_date();
        self.blurred.out_of_date();
        self.storage_data.out_of_date();
    }

    #[expect(unused)]
    pub fn resized(&mut self) {
        self.output.reset();
        self.storage_data.reset();
        self.storage_scratch.reset();
        self.blurred.reset();
    }

    pub fn zoomed(&mut self) {
        self.output.out_of_date();
        self.blurred.out_of_date();
        self.meta_buffer.out_of_date();
    }

    pub fn panned(&mut self) {
        self.output.out_of_date();
        self.meta_buffer.out_of_date();
    }

    pub fn cycled_filter(&mut self) {
        self.output.out_of_date();
        self.original.keep();
        self.params.keep();

        self.storage_layout.reset();
        self.kernel_layout.reset();
        self.convolution_layout.reset();
        self.convolution_pipeline.reset();

        self.copy_machine.reset();
        self.storage_data.reset();
        self.storage_scratch.reset();
        self.kernel_bind.reset();
        self.blurred.reset();

        self.texture_layout.reset();
        self.buffer_layout.reset();

        self.simple_pipeline_layout.reset();
        self.lanczos_pipeline_layout.reset();
        self.nearest_pipeline.reset();
        self.bilinear_pipeline.reset();
        self.lanczos_pipeline.reset();

        self.meta_buffer.keep();
        self.lanczos_buffer.reset();
    }
}
