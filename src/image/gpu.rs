use std::num::NonZeroU64;

use iced::wgpu;
use iced::wgpu::util::DeviceExt as _;

use super::ImageLoaded;
use crate::GpuContext;

const SHADER_ROOT: &str = "src/shader";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFilter {
    BiLinear,
    #[default]
    Nearest,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImagePipelineFeatures {
    pub filter: ImageFilter,
}

impl ImagePipelineFeatures {
    fn into_iter(self) -> impl Iterator<Item = (&'static str, bool)> {
        let mut features = vec![];

        match self.filter {
            ImageFilter::BiLinear => features.push(("FILTER_BILINEAR", true)),
            ImageFilter::Nearest => (),
        }

        features.into_iter()
    }
}

pub struct ImagePipeline {
    pipeline: wgpu::RenderPipeline,
    features: ImagePipelineFeatures,
    output_format: wgpu::TextureFormat,
}

impl ImagePipeline {
    const SHADER: &str = "package::image";

    pub fn new(
        ctx: &GpuContext,
        output_format: wgpu::TextureFormat,
        texture_layout: &TextureBindGroupLayout,
        meta_layout: &ImageMetadataBindGroupLayout,
        features: ImagePipelineFeatures,
    ) -> Self {
        let shader_module = &Self::SHADER.parse().expect("module path invalid");
        let shader_module = wesl::Wesl::new(SHADER_ROOT)
            .set_features(features.into_iter())
            .compile(shader_module)
            .inspect_err(|e| eprintln!("WESL error: {e}"))
            .expect("shader invalid")
            .to_string();
        let shader_module = wgpu::ShaderModuleDescriptor {
            label: Some("ColorNormalShader"),
            source: wgpu::ShaderSource::Wgsl(shader_module.into()),
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
            features,
            output_format,
        }
    }

    pub const fn output_format(&self) -> wgpu::TextureFormat {
        self.output_format
    }

    pub const fn features(&self) -> ImagePipelineFeatures {
        self.features
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
    size: wgpu::Extent3d,
}

impl ImageUploaded {
    pub const fn size(&self) -> wgpu::Extent3d {
        self.size
    }
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
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("image texture"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: image.format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
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

        Self { texture, size }
    }
}

pub struct Texture {
    _inner: wgpu::Texture,
    _view: wgpu::TextureView,
    _sampler: wgpu::Sampler,
    bind_group: wgpu::BindGroup,
}

impl Texture {
    pub const fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
}

pub struct ImageMetadataBinding {
    bind_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
}

impl ImageMetadataBinding {
    pub fn for_image(
        image: &ImageLoaded,
        ctx: &GpuContext,
        layout: &ImageMetadataBindGroupLayout,
    ) -> Self {
        #[expect(clippy::cast_precision_loss)]
        let data = ImageMetadataRaw {
            view_size: [image.width() as f32, image.height() as f32],
            image_size: [image.width() as f32, image.height() as f32],
            start: [0., 0.],
            scale: 1.,
            _pad: 0,
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
        Self { bind_group, buffer }
    }

    pub fn update(&self, ctx: &GpuContext, raw: &ImageMetadataRaw) {
        const SIZE: NonZeroU64 = NonZeroU64::new(std::mem::size_of::<ImageMetadataRaw>() as u64)
            .expect("struct not empty");

        let mut view = ctx
            .queue
            .write_buffer_with(&self.buffer, 0, SIZE)
            .expect("failed creating temporary buffer for upload");

        view.copy_from_slice(bytemuck::cast_slice(&[*raw]));
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
    pub view_size: [f32; 2],
    /// (x, y) of the top left corner
    pub image_size: [f32; 2],
    /// (width, height) of the visible area
    pub start: [f32; 2],
    /// scale of image
    pub scale: f32,
    /// padding to get to a multiple of alignment bytes (8)
    pub _pad: u32,
}
