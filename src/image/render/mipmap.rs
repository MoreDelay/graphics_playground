use core::f32;

use iced::wgpu;

use crate::image::filters::GaussFilter;
use crate::instruments::GpuContext;
use crate::instruments::bind::storage::{SimpleStorageTexture, StorageTextureCopyMachine};
use crate::instruments::pipeline::filter::{
    ConvolutionPipeline,
    ConvolutionPipelineLayout,
    KernelBinding,
    KernelLayout,
    StorageSrcDstLayout,
};

pub struct MipMapper {
    halfing: wgpu::ComputePipeline,
    convolution: ConvolutionPipeline,

    storage_layout: StorageSrcDstLayout,
    #[expect(unused)]
    kernel_layout: KernelLayout,
}

impl MipMapper {
    const SHADER_HALFING: &str = "package::mipmap::halfing";

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
        storage_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::ComputePipeline {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("MipMapper Halfing Pipeline Layout"),
                bind_group_layouts: &[storage_layout],
                push_constant_ranges: &[],
            });
        let module = crate::instruments::create_simple_shader_module_desc(
            Some("MipMapper Halfing Shader"),
            Self::SHADER_HALFING,
        );
        let module = ctx.device.create_shader_module(module);
        ctx.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MipMapper Halfing Pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("halfing"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
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
        ConvolutionPipeline::new(ctx, &layout, Some("MipMapper Convolution Pipeline"))
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
                self.run_halfing(ctx, &mut pass, mip_level);
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

    fn run_halfing(&self, ctx: &GpuContext, pass: &mut wgpu::ComputePass, source_mip_level: u32) {
        let target_mip_level = source_mip_level + 1;

        let src_view = self
            .texture_filtered_2d
            .create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: source_mip_level,
                mip_level_count: Some(1),
                ..Default::default()
            });
        let dst_view = self
            .texture_downsampled
            .create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: target_mip_level,
                mip_level_count: Some(1),
                ..Default::default()
            });
        let texture_bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MipMapper Halfing BindGroup"),
            layout: &self.mip_mapper.storage_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&dst_view),
                },
            ],
        });

        // divide by 2^mip_level
        let dispatch_x = self.texture_downsampled.width() >> target_mip_level;
        let dispatch_y = self.texture_downsampled.height() >> target_mip_level;
        let dispatch_x = dispatch_x.div_ceil(16);
        let dispatch_y = dispatch_y.div_ceil(16);

        pass.set_pipeline(&self.mip_mapper.halfing);
        pass.set_bind_group(0, &texture_bind_group, &[]);
        pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
    }
}
