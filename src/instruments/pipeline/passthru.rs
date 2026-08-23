//! Contains the Pass-Thru pipeline that acts as a simple copy between textures

use iced::wgpu;

use crate::instruments::GpuContext;

/// A render pipeline that just passes renders a texture to the output unchanged
pub struct PassThruPipeline {
    /// The inner render pipeline
    pipeline: wgpu::RenderPipeline,
    /// The layout for the source texture
    texture_layout: PassThruTextureLayout,
    /// The color format for the output target
    output_format: wgpu::TextureFormat,
}

impl PassThruPipeline {
    /// Pass-thru vertex shader path
    const SHADER_VERTEX_QUAD: &str = "package::image::quad";
    /// Pass-thru fragment shader path
    const SHADER_FRAGMENT_PASSTHRU: &str = "package::passthru";

    /// Create a new pipeline
    ///
    /// When drawing with this, the output target must match the output format.
    pub fn new(ctx: &GpuContext, output_format: wgpu::TextureFormat) -> Self {
        let vs_module = crate::instruments::create_simple_shader_module_desc(
            Some("Quad Shader"),
            Self::SHADER_VERTEX_QUAD,
        );
        let vs_module = ctx.device.create_shader_module(vs_module);

        let fs_module = crate::instruments::create_simple_shader_module_desc(
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

    /// Create a new rendering target compatible with this pipeline
    pub fn create_texture(&self, ctx: &GpuContext, size: wgpu::Extent3d) -> PassThruTexture {
        PassThruTexture::new(ctx, &self.texture_layout, size, self.output_format)
    }

    /// Draw a rendering to the target of the same size
    pub fn full_draw(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        rendering: &PassThruTexture,
        target: &wgpu::TextureView,
    ) {
        let in_size = rendering.texture().size();
        let out_size = target.texture().size();
        assert_eq!(
            in_size, out_size,
            "input texture does not match output size"
        );

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Viewport PassThru Render Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load, // iced drew the gui already, so load that
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        self.draw(&mut pass, rendering);
    }

    /// Draw a rendering using the provided pass
    ///
    /// Prefer [`Self::full_draw`] when the render pass does not need special handling
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>, rendering: &PassThruTexture) {
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, rendering.bind(), &[]);
        pass.draw(0..4, 0..1);
    }
}

/// Contains the handles for a texture used with the [`PassThruPipeline`]
#[derive(Clone)]
pub struct PassThruTexture {
    /// The inner texture
    texture: wgpu::Texture,
    /// A view into this texture
    ///
    /// Also used in the binding
    view: wgpu::TextureView,
    /// The binding of this texture
    bind: wgpu::BindGroup,
}

impl PassThruTexture {
    /// Create a new texture
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

    /// Get a view to this texture
    pub const fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Get the bind group for this texture
    pub const fn bind(&self) -> &wgpu::BindGroup {
        &self.bind
    }

    /// Get the inner texture
    pub const fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

/// The bind group layout for a [`PassThruTexture`]
struct PassThruTextureLayout {
    /// The inner layout
    layout: wgpu::BindGroupLayout,
    /// The sampler used in the pass-thru shader
    sampler: wgpu::Sampler,
}

impl PassThruTextureLayout {
    /// Create a new layout
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
