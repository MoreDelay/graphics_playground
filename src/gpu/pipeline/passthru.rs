use iced::wgpu;

use crate::gpu::GpuContext;

pub struct PassThruPipeline {
    pipeline: wgpu::RenderPipeline,
    texture_layout: PassThruTextureLayout,
    output_format: wgpu::TextureFormat,
}

impl PassThruPipeline {
    const SHADER_VERTEX_QUAD: &str = "package::image::quad";
    const SHADER_FRAGMENT_PASSTHRU: &str = "package::passthru";

    pub fn new(ctx: &GpuContext, output_format: wgpu::TextureFormat) -> Self {
        let vs_module = crate::gpu::create_simple_shader_module_desc(
            Some("Quad Shader"),
            Self::SHADER_VERTEX_QUAD,
        );
        let vs_module = ctx.device.create_shader_module(vs_module);

        let fs_module = crate::gpu::create_simple_shader_module_desc(
            Some("PassThru Fragment Shader"),
            Self::SHADER_FRAGMENT_PASSTHRU,
        );
        let fs_module = ctx.device.create_shader_module(fs_module);

        let texture_layout = PassThruTextureLayout::new(ctx);
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PassThru Pipeline Layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[&texture_layout.layout],
            });

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("PassThru Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vs_module,
                    entry_point: Some("vs_quad"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs_module,
                    entry_point: Some("fs_passthru"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: output_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    front_face: wgpu::FrontFace::Ccw,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        Self {
            pipeline,
            texture_layout,
            output_format,
        }
    }

    pub fn create_texture(&self, ctx: &GpuContext, size: wgpu::Extent3d) -> PassThruTexture {
        PassThruTexture::new(ctx, &self.texture_layout, size, self.output_format)
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, texture: &PassThruTexture) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, texture.bind(), &[]);
        pass.draw(0..4, 0..1);
    }
}

pub struct PassThruTexture {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    bind: wgpu::BindGroup,
}

impl PassThruTexture {
    fn new(
        ctx: &GpuContext,
        layout: &PassThruTextureLayout,
        size: wgpu::Extent3d,
        format: wgpu::TextureFormat,
    ) -> Self {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("PassThru Texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PassThru Texture Bind Group"),
            layout: &layout.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&layout.sampler),
                },
            ],
        });

        Self {
            texture,
            view,
            bind,
        }
    }

    pub const fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    pub const fn bind(&self) -> &wgpu::BindGroup {
        &self.bind
    }

    pub const fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

struct PassThruTextureLayout {
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
}

impl PassThruTextureLayout {
    fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PassThru Texture Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("PassThru Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..wgpu::SamplerDescriptor::default()
        });

        Self { layout, sampler }
    }
}
