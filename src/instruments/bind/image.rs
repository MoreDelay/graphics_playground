use iced::wgpu;

use crate::instruments::GpuContext;
use crate::instruments::buffer::SimpleBuffer;

pub struct ImageMetadataLayout(wgpu::BindGroupLayout);

impl ImageMetadataLayout {
    pub fn new(ctx: &GpuContext) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Image Metadata Bind Group Layout"),
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
        meta: &SimpleBuffer<ImageMetadataRaw>,
    ) -> Self {
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Image Metadata Bind Group"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: meta.resource(),
            }],
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
pub struct ImageMetadataRaw {
    /// (width, height) of the visible area
    pub start: [f32; 2],
    /// zoom of image (greater than 1 means magnification)
    pub zoom: f32,
    /// padding to get to a multiple of alignment bytes (8)
    pub _pad: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LanczosInfoRaw {
    /// Size of windowing function, typically 2 or 3
    pub filter_size: f32,
}
