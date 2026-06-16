use std::path::Path;

use iced::wgpu::util::DeviceExt as _;
use iced::wgpu::{self};

use crate::{GpuContext, TargetContext};

pub struct ImageRenderState {
    image: ImageUploaded,
    _texture_layout: TextureBindGroupLayout,
    _meta_layout: ImageMetadataBindGroupLayout,
    meta_bind: ImageMetadataBinding,
    pipeline: ImagePipeline,
}

impl ImageRenderState {
    pub fn new(image: &ImageLoaded, ctx: &GpuContext, target: &TargetContext) -> Self {
        let texture_layout = TextureBindGroupLayout::new(ctx);
        let meta_layout = ImageMetadataBindGroupLayout::new(ctx);
        let uploaded = image.upload(ctx, &texture_layout);
        let meta_bind = ImageMetadataBinding::for_image(image, ctx, &meta_layout);
        let pipeline = ImagePipeline::new(ctx, target, &texture_layout, &meta_layout);
        Self {
            image: uploaded,
            _texture_layout: texture_layout,
            _meta_layout: meta_layout,
            meta_bind,
            pipeline,
        }
    }
}

impl ImageRenderState {
    pub fn draw<'a>(&'a self, render_pass: &mut wgpu::RenderPass<'a>) {
        render_pass.set_pipeline(&self.pipeline);
        render_pass.set_bind_group(0, &self.image.bind_group, &[]);
        render_pass.set_bind_group(1, &*self.meta_bind, &[]);
        render_pass.draw(0..4, 0..1);
    }
}

struct Texture {
    _inner: wgpu::Texture,
    _view: wgpu::TextureView,
    _sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
}

pub struct TextureBindGroupLayout(wgpu::BindGroupLayout);

impl TextureBindGroupLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Texture Bind Group Layout"),
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
        Self(layout)
    }
}

impl std::ops::Deref for TextureBindGroupLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ImageLoaded {
    image: image::RgbaImage,
    format: wgpu::TextureFormat,
}

impl std::ops::Deref for ImageLoaded {
    type Target = image::RgbaImage;

    fn deref(&self) -> &Self::Target {
        &self.image
    }
}

impl ImageLoaded {
    pub fn load(path: &Path, format: wgpu::TextureFormat) -> Result<Self, image::ImageError> {
        let image = image::ImageReader::open(path)?
            .with_guessed_format()?
            .decode()?;
        let image = image.into();
        Ok(Self { image, format })
    }

    fn upload(&self, ctx: &GpuContext, layout: &TextureBindGroupLayout) -> ImageUploaded {
        let size = wgpu::Extent3d {
            width: self.image.width(),
            height: self.image.height(),
            depth_or_array_layers: 1,
        };
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            // specified format above supported by default, only additional view formats here
            view_formats: &[],
        });

        // assuming only uncompressed formats are used here
        let texel_bytes = self
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
            &self.image,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(texel_bytes * size.width),
                rows_per_image: Some(size.height),
            },
            size,
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("texture sampler"),
            address_mode_u: wgpu::AddressMode::ClampToBorder,
            address_mode_v: wgpu::AddressMode::ClampToBorder,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::FilterMode::Nearest,
            border_color: Some(wgpu::SamplerBorderColor::TransparentBlack),
            ..Default::default()
        });
        let entries = [
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&sampler),
            },
        ];
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Texture Bind Group"),
            layout,
            entries: &entries,
        });
        let texture = Texture {
            _inner: texture,
            _view: view,
            _sampler: sampler,
            bind_group,
        };

        ImageUploaded {
            texture,
            _size: size,
        }
    }
}

struct ImageUploaded {
    texture: Texture,
    _size: wgpu::Extent3d,
}

impl std::ops::Deref for ImageUploaded {
    type Target = Texture;

    fn deref(&self) -> &Self::Target {
        &self.texture
    }
}

struct ImageMetadataBinding {
    bind_group: wgpu::BindGroup,
    _buffer: wgpu::Buffer,
}

impl std::ops::Deref for ImageMetadataBinding {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.bind_group
    }
}

impl ImageMetadataBinding {
    fn for_image(
        image: &ImageLoaded,
        ctx: &GpuContext,
        layout: &ImageMetadataBindGroupLayout,
    ) -> Self {
        #[expect(clippy::cast_precision_loss)]
        let data = ImageMetadataRaw {
            size: [image.width() as f32, image.height() as f32],
            start: [0., 0.],
            area: [image.width() as f32, image.height() as f32],
        };
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ImageMetadata Buffer"),
                contents: bytemuck::cast_slice(&[data]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ImageMetadata Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.as_entire_binding(),
            }],
        });
        Self {
            bind_group,
            _buffer: buffer,
        }
    }
}

pub struct ImageMetadataBindGroupLayout(wgpu::BindGroupLayout);

impl ImageMetadataBindGroupLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ImageMetadata Bind Group Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
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
pub struct ImageMetadataRaw {
    /// (width, height) of the whole image
    pub size: [f32; 2],
    /// (x, y) of the top left corner
    pub start: [f32; 2],
    /// (width, height) of the visible area
    pub area: [f32; 2],
}

struct ImagePipeline {
    pipeline: wgpu::RenderPipeline,
}

impl std::ops::Deref for ImagePipeline {
    type Target = wgpu::RenderPipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

impl ImagePipeline {
    // const SHADER: &str = "package::image";
    const SHADER_TEXT: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/shader/image.wgsl"
    ));

    fn new(
        ctx: &GpuContext,
        target: &TargetContext,
        texture_layout: &TextureBindGroupLayout,
        meta_layout: &ImageMetadataBindGroupLayout,
    ) -> Self {
        // let shader = wesl::Wesl::new(Self::SHADER)
        //     .compile(&Self::ENTRY.parse().unwrap())
        //     .inspect_err(|e| eprintln!("WESL error: {e}"))
        //     .unwrap()
        //     .to_string();
        let shader_module = wgpu::ShaderModuleDescriptor {
            label: Some("ColorNormalShader"),
            source: wgpu::ShaderSource::Wgsl(Self::SHADER_TEXT.into()),
        };
        let shader_module = ctx.device.create_shader_module(shader_module);

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
                    module: &shader_module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader_module,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: target.config.format,
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

        Self { pipeline }
    }
}
