use core::f32;

use iced::wgpu;

use crate::image::filters::GaussFilter;
use crate::instruments::GpuContext;
use crate::instruments::bind::storage::{
    SimpleStorageTexture,
    StorageSrcDstLayout,
    StorageTextureCopyMachine,
};
use crate::instruments::pipeline::filter::{
    ConvolutionPipeline,
    ConvolutionPipelineLayout,
    KernelBinding,
    KernelLayout,
};
use crate::instruments::pipeline::halfing::{HalfingPipeline, HalfingPipelineLayout};

pub struct MipMapper {
    halfing: HalfingPipeline,
    convolution: ConvolutionPipeline,

    storage_layout: StorageSrcDstLayout,
    #[expect(unused)]
    kernel_layout: KernelLayout,
}

impl MipMapper {
    pub fn new(ctx: &GpuContext) -> Self {
        let storage_layout =
            StorageSrcDstLayout::new(ctx, Some("MipMapper Storage Texture Layout"));
        let kernel_layout = KernelLayout::new(ctx, Some("MipMapper Kernel Texture Layout"));

        let halfing = Self::create_pipeline_halfing(ctx, &storage_layout);
        let convolution = Self::create_pipeline_convolution(ctx, &storage_layout, &kernel_layout);

        Self {
            halfing,
            convolution,
            storage_layout,
            kernel_layout,
        }
    }

    pub fn compute_mipmaps(&self, ctx: &GpuContext, texture: &wgpu::Texture) {
        assert!(
            texture.format().is_srgb(),
            "expect sRGB textures (due to copy pipeline)"
        );

        let Some(runner) = MipMapRunner::new(ctx, self, texture) else {
            return;
        };
        runner.run(ctx);
    }

    fn create_pipeline_halfing(
        ctx: &GpuContext,
        storage_layout: &StorageSrcDstLayout,
    ) -> HalfingPipeline {
        let layout = HalfingPipelineLayout::new(
            ctx,
            storage_layout,
            Some("MipMapper Halfing Pipeline Layout"),
        );
        HalfingPipeline::new(
            ctx,
            &layout,
            Some("MipMapper Halfing Shader"),
            Some("MipMapper Halfing Pipeline"),
        )
    }

    fn create_pipeline_convolution(
        ctx: &GpuContext,
        storage_layout: &StorageSrcDstLayout,
        kernel_layout: &KernelLayout,
    ) -> ConvolutionPipeline {
        let layout = ConvolutionPipelineLayout::new(
            ctx,
            storage_layout,
            kernel_layout,
            Some("MipMapper Convolution Pipeline Layout"),
        );
        ConvolutionPipeline::new(
            ctx,
            &layout,
            Some("MipMapper Convolution Shader"),
            Some("MipMapper Convolution Pipeline"),
        )
    }
}

struct MipMapRunner<'a> {
    mip_mapper: &'a MipMapper,
    texture: &'a wgpu::Texture,
    copy_helper: StorageTextureCopyMachine,
    texture_filtered_1d: SimpleStorageTexture,
    texture_filtered_2d: SimpleStorageTexture,
    texture_downsampled: SimpleStorageTexture,
    kernel_bind: KernelBinding,
}

impl<'a> MipMapRunner<'a> {
    fn new(
        ctx: &GpuContext,
        mip_mapper: &'a MipMapper,
        texture: &'a wgpu::Texture,
    ) -> Option<Self> {
        use wgpu::TextureFormat::*;

        const SIGMA: f32 = 0.5;

        assert!(
            texture.format() == Rgba8UnormSrgb,
            "only handling this format atm due to copy"
        );

        if texture.mip_level_count() == 1 {
            return None;
        }

        let label = Some("MipMapper filtered-1d storage texture");
        let texture_filtered_1d = SimpleStorageTexture::empty(ctx, texture, label);

        let label = Some("MipMapper filtered-2d storage texture");
        let texture_filtered_2d = SimpleStorageTexture::empty(ctx, texture, label);

        let label = Some("MipMapper downsampled storage texture");
        let texture_downsampled = SimpleStorageTexture::empty(ctx, texture, label);

        let kernel_layout = KernelLayout::new(ctx, Some("MipMapper Kernel Layout"));
        let kernel = &GaussFilter::new(SIGMA).expect("valid sigma").blur_kernel();
        let kernel_bind =
            KernelBinding::new(ctx, &kernel_layout, kernel, Some("MipMapper Kernel Bind"));

        let copy_helper = StorageTextureCopyMachine::new(ctx, texture.format());

        Some(Self {
            mip_mapper,
            texture,
            copy_helper,
            texture_filtered_1d,
            texture_filtered_2d,
            texture_downsampled,
            kernel_bind,
        })
    }

    fn run(self, ctx: &GpuContext) {
        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Mipmap Command Encoder"),
            });

        // copy over start texture to base level in downsampled texture stack
        self.copy_helper.to_storage(
            ctx,
            &mut encoder,
            self.texture,
            &self.texture_downsampled,
            0,
        );

        // run mip map construction
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MipMapper Compute Pass"),
                timestamp_writes: None,
            });

            let mip_level_iterations = self.texture.mip_level_count() - 1;
            for mip_level in 0..mip_level_iterations {
                self.mip_mapper.convolution.run(
                    ctx,
                    &mut pass,
                    &self.mip_mapper.storage_layout,
                    &self.texture_downsampled,
                    &self.texture_filtered_1d,
                    &self.texture_filtered_2d,
                    &self.kernel_bind,
                    mip_level,
                );
                self.mip_mapper.halfing.run(
                    ctx,
                    &mut pass,
                    &self.mip_mapper.storage_layout,
                    &self.texture_filtered_2d,
                    &self.texture_downsampled,
                    mip_level,
                );
            }
        }

        // copy computed mip maps over to texture
        for mip_level in 1..self.texture.mip_level_count() {
            self.copy_helper.to_texture(
                ctx,
                &mut encoder,
                &self.texture_downsampled,
                self.texture,
                mip_level,
            );
        }

        ctx.queue.submit([encoder.finish()]);
    }
}
