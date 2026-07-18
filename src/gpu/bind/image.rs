use iced::wgpu;

use crate::gpu::{GpuContext, SimpleBuffer};

pub struct SingleTextureLayout(wgpu::BindGroupLayout);

impl SingleTextureLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Single Texture Bind Group Layout"),
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

impl std::ops::Deref for SingleTextureLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct SingleTextureBind(wgpu::BindGroup);

impl SingleTextureBind {
    pub fn new(ctx: &GpuContext, layout: &SingleTextureLayout, texture: &wgpu::Texture) -> Self {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Single Texture Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        Self(bind)
    }
}

impl std::ops::Deref for SingleTextureBind {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ImageMetadataLayout(wgpu::BindGroupLayout);

impl ImageMetadataLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Image Metadata Bind Group Layout"),
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

impl std::ops::Deref for ImageMetadataLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub struct ImageMetadataBind(wgpu::BindGroup);

impl ImageMetadataBind {
    pub fn new(
        ctx: &GpuContext,
        layout: &ImageMetadataLayout,
        viewport: &SimpleBuffer<ViewportRaw>,
        meta: &SimpleBuffer<ImageMetadataRaw>,
    ) -> Self {
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Image Metadata Bind Group"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: viewport.resource(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: meta.resource(),
                },
            ],
        });
        Self(bind)
    }
}

impl std::ops::Deref for ImageMetadataBind {
    type Target = wgpu::BindGroup;

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
