use iced::wgpu;

use crate::GpuContext;
use crate::gpu::SimpleBuffer;
use crate::image::ImageLoaded;

pub struct Interpolator {
    #[expect(unused)]
    meta_layout: wgpu::BindGroupLayout,
    image_layout: wgpu::BindGroupLayout,
    #[expect(unused)]
    lanczos_buffer: SimpleBuffer<LanczosInfoRaw>,
    image_buffer: SimpleBuffer<ImageInfoRaw>,
    meta_bind: wgpu::BindGroup,
    pipeline_interpolate: wgpu::RenderPipeline,
}

impl Interpolator {
    const SHADER_VERTEX: &str = "package::image::quad";
    const SHADER_FRAGMENT: &str = "package::image::lanczos";

    pub fn new(ctx: &GpuContext) -> Self {
        let meta_layout = Self::create_meta_layout(ctx);
        let image_layout = Self::create_image_layout(ctx);
        let pipeline_interpolate =
            Self::create_pipeline_interpolate(ctx, &meta_layout, &image_layout);

        let contents = LanczosInfoRaw { filter_size: 1. };
        let lanczos_buffer = SimpleBuffer::new(ctx, contents, Some("Lanczos Info Buffer"));
        let contents = ImageInfoRaw { new_size: [1, 1] };
        let image_buffer = SimpleBuffer::new(ctx, contents, Some("Lanczos Image Info Buffer"));
        let meta_bind = Self::create_meta_bind(ctx, &meta_layout, &lanczos_buffer, &image_buffer);

        Self {
            meta_layout,
            image_layout,
            lanczos_buffer,
            image_buffer,
            meta_bind,
            pipeline_interpolate,
        }
    }

    pub fn filter(&self, ctx: &GpuContext, src: &wgpu::Texture, dst: &wgpu::Texture) {
        assert!(
            src.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING),
            "src is used as texture binding here"
        );
        assert!(
            dst.usage().contains(wgpu::TextureUsages::RENDER_ATTACHMENT),
            "dst is used as render target here",
        );

        {
            let image_raw = ImageInfoRaw {
                new_size: [dst.width(), dst.height()],
            };
            self.image_buffer.update(ctx, image_raw);
        }

        let src_view = src.create_view(&wgpu::TextureViewDescriptor::default());
        let dst_view = dst.create_view(&wgpu::TextureViewDescriptor::default());

        let src_bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lanczos Source Image Bind Group"),
            layout: &self.image_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&src_view),
            }],
        });

        let mut encoder = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Lanczos Interpolation Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &dst_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..wgpu::RenderPassDescriptor::default()
            });

            pass.set_pipeline(&self.pipeline_interpolate);
            pass.set_bind_group(0, &self.meta_bind, &[]);
            pass.set_bind_group(1, &src_bind, &[]);
            pass.draw(0..4, 0..1);
        }

        ctx.queue.submit([encoder.finish()]);
    }

    fn create_meta_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lanczos Metadata Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            })
    }

    fn create_image_layout(ctx: &GpuContext) -> wgpu::BindGroupLayout {
        ctx.device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Lanczos Pipeline Lanczos Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            })
    }

    fn create_meta_bind(
        ctx: &GpuContext,
        layout: &wgpu::BindGroupLayout,
        lanczos_buffer: &SimpleBuffer<LanczosInfoRaw>,
        image_buffer: &SimpleBuffer<ImageInfoRaw>,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Lanczos Meta Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lanczos_buffer.resource(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_buffer.resource(),
                },
            ],
        })
    }

    fn create_pipeline_interpolate(
        ctx: &GpuContext,
        lanczos_layout: &wgpu::BindGroupLayout,
        image_layout: &wgpu::BindGroupLayout,
    ) -> wgpu::RenderPipeline {
        let layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Lanczos Pipeline Layout"),
                bind_group_layouts: &[lanczos_layout, image_layout],
                push_constant_ranges: &[],
            });
        let vs_module = crate::gpu::create_simple_shader_module_desc(
            Some("Lanczos Quad Vertex Shader"),
            Self::SHADER_VERTEX,
        );
        let vs_module = ctx.device.create_shader_module(vs_module);
        let fs_module = crate::gpu::create_simple_shader_module_desc(
            Some("Lanczos Interpolation Fragment Shader"),
            Self::SHADER_FRAGMENT,
        );
        let fs_module = ctx.device.create_shader_module(fs_module);
        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Lanczos Pipeline"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &vs_module,
                    entry_point: Some("vs_quad"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..wgpu::PrimitiveState::default()
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs_module,
                    entry_point: Some("fs_lanczos"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: ImageLoaded::FORMAT_SRGB,
                        blend: None,
                        write_mask: wgpu::ColorWrites::default(),
                    })],
                }),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LanczosInfoRaw {
    filter_size: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ImageInfoRaw {
    new_size: [u32; 2],
}
