use iced::wgpu;

use crate::instruments::GpuContext;

pub struct SimpleTexture {
    texture: wgpu::Texture,
    #[expect(unused)]
    view: wgpu::TextureView,
    bind: wgpu::BindGroup,
}

impl SimpleTexture {
    pub fn new(
        ctx: &GpuContext,
        layout: &SimpleTextureLayout,
        texture: wgpu::Texture,
        label: Option<&str>,
    ) -> Self {
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        Self {
            texture,
            view,
            bind,
        }
    }

    pub fn empty(
        ctx: &GpuContext,
        layout: &SimpleTextureLayout,
        base: &wgpu::Texture,
        label: Option<&str>,
    ) -> Self {
        let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
            label,
            size: base.size(),
            mip_level_count: base.mip_level_count(),
            sample_count: base.sample_count(),
            dimension: base.dimension(),
            format: base.format(),
            usage: base.usage(),
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&view),
            }],
        });
        Self {
            texture,
            view,
            bind,
        }
    }

    pub const fn texture(&self) -> &wgpu::Texture {
        &self.texture
    }
}

impl std::ops::Deref for SimpleTexture {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.bind
    }
}

pub struct SimpleTextureLayout(wgpu::BindGroupLayout);

impl SimpleTextureLayout {
    pub fn new(ctx: &GpuContext, label: Option<&str>) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label,
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

impl std::ops::Deref for SimpleTextureLayout {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
