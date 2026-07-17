pub mod lanczos;
mod mipmap;

use iced::wgpu;

use super::ImageLoaded;
use crate::GpuContext;
use crate::gpu::SimpleBuffer;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFilter {
    #[default]
    Nearest,
    BiLinear,
    Lanczos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImagePipelineFeatures {
    pub filter: ImageFilter,
}

impl ImagePipelineFeatures {
    fn into_iter(self) -> impl Iterator<Item = (&'static str, bool)> {
        let mut features = vec![];

        match self.filter {
            ImageFilter::Nearest => (),
            ImageFilter::BiLinear => features.push(("FILTER_BILINEAR", true)),
            ImageFilter::Lanczos => (),
        }

        features.into_iter()
    }
}

pub struct ImagePipeline {
    pipeline: wgpu::RenderPipeline,
    output_format: wgpu::TextureFormat,
}

impl ImagePipeline {
    const SHADER_VERTEX: &str = "package::image::quad";
    const SHADER_FRAGMENT: &str = "package::image::render";

    pub fn new(
        ctx: &GpuContext,
        output_format: wgpu::TextureFormat,
        texture_layout: &TextureBindGroupLayout,
        meta_layout: &ImageMetadataBindGroupLayout,
        features: ImagePipelineFeatures,
    ) -> Self {
        let vs_module =
            crate::gpu::create_simple_shader_module_desc(Some("Quad Shader"), Self::SHADER_VERTEX);
        let vs_module = ctx.device.create_shader_module(vs_module);

        let fs_module = &Self::SHADER_FRAGMENT.parse().expect("module path invalid");
        let fs_module = wesl::Wesl::new(crate::gpu::SHADER_ROOT)
            .set_features(features.into_iter())
            .compile(fs_module)
            .inspect_err(|e| eprintln!("WESL error: {e}"))
            .expect("shader invalid")
            .to_string();
        let fs_module = wgpu::ShaderModuleDescriptor {
            label: Some("ColorNormalShader"),
            source: wgpu::ShaderSource::Wgsl(fs_module.into()),
        };
        let fs_module = ctx.device.create_shader_module(fs_module);

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Image Pipeline Layout"),
                push_constant_ranges: &[],
                bind_group_layouts: &[texture_layout, meta_layout],
            });

        let pipeline = ctx
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("Image Pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &vs_module,
                    entry_point: Some("vs_quad"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fs_module,
                    entry_point: Some("fs_image"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: output_format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
                multisample: wgpu::MultisampleState {
                    count: 1,
                    mask: !0,
                    alpha_to_coverage_enabled: false,
                },
                multiview: None,
                cache: None,
            });

        Self {
            pipeline,
            output_format,
        }
    }

    pub const fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }
}

impl std::ops::Deref for ImagePipeline {
    type Target = wgpu::RenderPipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

pub struct ImageUploaded {
    texture: Texture,
}

impl std::ops::Deref for ImageUploaded {
    type Target = Texture;

    fn deref(&self) -> &Self::Target {
        &self.texture
    }
}

impl ImageUploaded {
    pub fn upload(image: &ImageLoaded, ctx: &GpuContext, layout: &TextureBindGroupLayout) -> Self {
        let size = wgpu::Extent3d {
            width: image.width(),
            height: image.height(),
            depth_or_array_layers: 1,
        };

        let mip_level_count = size.width.min(size.height).ilog2() + 1;
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image texture"),
            size,
            mip_level_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: image.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            // specified format above supported by default, only additional view formats here
            view_formats: &[],
        });

        // assuming only uncompressed formats are used here
        let texel_bytes = image
            .format
            .block_copy_size(None)
            .expect("assuming no complex texture format is used");

        // load image (on CPU) into texture (on GPU) by issuing command over queue
        ctx.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            image,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(texel_bytes * size.width),
                rows_per_image: Some(size.height),
            },
            size,
        );

        let mipmapper = mipmap::MipMapper::new(ctx);
        mipmapper.compute_mipmaps(ctx, &texture);

        let texture = Texture::new(ctx, texture, layout);

        Self { texture }
    }

    pub fn texture(&self) -> &wgpu::Texture {
        &self.inner
    }
}

pub struct Texture {
    inner: wgpu::Texture,
    _view: wgpu::TextureView,
    bind_group: wgpu::BindGroup,
}

impl Texture {
    pub fn new(ctx: &GpuContext, texture: wgpu::Texture, layout: &TextureBindGroupLayout) -> Self {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        Self {
            inner: texture,
            _view: view,
            bind_group,
        }
    }

    pub const fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

pub struct ImageMetadataBinding {
    bind_group: wgpu::BindGroup,
    viewport: SimpleBuffer<ViewportRaw>,
    image_meta: SimpleBuffer<ImageMetadataRaw>,
}

impl ImageMetadataBinding {
    pub fn new(ctx: &GpuContext, layout: &ImageMetadataBindGroupLayout) -> Self {
        let viewport = ViewportRaw {
            origin: [0, 0],
            size: [1, 1],
            scale: 1.,
            _pad: 0,
        };
        let viewport = SimpleBuffer::new(ctx, viewport, Some("Viewport Buffer"));

        let image_meta = ImageMetadataRaw {
            start: [0., 0.],
            zoom: 1.,
            _pad: 0,
        };
        let image_meta = SimpleBuffer::new(ctx, image_meta, Some("ImageMetadata Buffer"));

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Metadata Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: viewport.resource(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: image_meta.resource(),
                },
            ],
        });
        Self {
            bind_group,
            viewport,
            image_meta,
        }
    }

    pub fn update_viewport(&self, ctx: &GpuContext, data: ViewportRaw) {
        self.viewport.update(ctx, data);
    }

    pub fn update_image_metadata(&self, ctx: &GpuContext, data: ImageMetadataRaw) {
        self.image_meta.update(ctx, data);
    }
}

impl std::ops::Deref for ImageMetadataBinding {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.bind_group
    }
}

pub struct TextureBindGroupLayout(wgpu::BindGroupLayout);

impl TextureBindGroupLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Texture Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                }],
            });
        Self(layout)
    }
}

impl std::ops::Deref for TextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ImageMetadataBindGroupLayout(wgpu::BindGroupLayout);

impl ImageMetadataBindGroupLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ImageMetadata Bind Group Layout"),
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
            });
        Self(layout)
    }
}

impl std::ops::Deref for ImageMetadataBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewportRaw {
    /// (x, y) in framebuffer where viewport starts (top-left corner)
    pub origin: [u32; 2],
    /// (width, height) of the viewport
    pub size: [u32; 2],
    /// scale factor of monitor
    pub scale: f32,
    /// padding to get to a multiple of alignment bytes (8)
    pub _pad: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ImageMetadataRaw {
    /// (width, height) of the visible area
    pub start: [f32; 2],
    /// zoom of image (greater than 1 means magnification)
    pub zoom: f32,
    /// padding to get to a multiple of alignment bytes (8)
    pub _pad: u32,
}
