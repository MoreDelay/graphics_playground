//! Instruments for uniform buffers

use std::marker::PhantomData;
use std::num::NonZeroU64;

use iced::wgpu;
use iced::wgpu::util::DeviceExt as _;

use crate::instruments::GpuContext;

/// A marker trait for types that can be used as raw data provides for uniform buffers
pub trait BufferRaw: Copy + Clone + bytemuck::Pod + bytemuck::Zeroable {}
impl<T> BufferRaw for T where T: Copy + Clone + bytemuck::Pod + bytemuck::Zeroable {}

/// A simple uniform buffer
pub struct SimpleBuffer<R: BufferRaw> {
    /// The gpu buffer
    buffer: wgpu::Buffer,
    /// Marker to associate this buffer with some data provider type
    marker: PhantomData<R>,
}

impl<R: BufferRaw> SimpleBuffer<R> {
    /// The number of bytes stored in this buffer
    const SIZE: NonZeroU64 =
        NonZeroU64::new(std::mem::size_of::<R>() as u64).expect("struct not empty");

    /// Create a new buffer
    pub fn new(ctx: &GpuContext, init: R, label: Option<&str>) -> Self {
        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label,
                contents: bytemuck::cast_slice(&[init]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let marker = PhantomData;
        Self { buffer, marker }
    }

    /// Update the contents of this buffer
    pub fn update(&self, ctx: &GpuContext, data: R) {
        ctx.queue
            .write_buffer_with(&self.buffer, 0, Self::SIZE)
            .expect("failed creating temporary buffer for upload")
            .copy_from_slice(bytemuck::cast_slice(&[data]));
    }

    /// Get this buffer as a [`wgpu::BindingResource`]
    pub fn resource(&self) -> wgpu::BindingResource<'_> {
        self.buffer.as_entire_binding()
    }
}

/// A bind group for a uniform buffer
pub struct SimpleBufferBind<R: BufferRaw, V: Visibility> {
    /// The buffer used in the binding
    buffer: SimpleBuffer<R>,
    /// The bind group
    bind: wgpu::BindGroup,
    /// Marker for visibility
    _marker: PhantomData<V>,
}

impl<R: BufferRaw, V: Visibility> SimpleBufferBind<R, V> {
    /// Create a new binding
    pub fn new(
        ctx: &GpuContext,
        buffer: SimpleBuffer<R>,
        layout: &SimpleBufferBindLayout<V>,
        label: Option<&str>,
    ) -> Self {
        let bind = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label,
            layout: &layout.0,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: buffer.resource(),
            }],
        });

        Self {
            buffer,
            bind,
            _marker: PhantomData,
        }
    }

    /// Access the buffer of this binding
    pub const fn buffer(&self) -> &SimpleBuffer<R> {
        &self.buffer
    }
}

impl<R: BufferRaw, V: Visibility> std::ops::Deref for SimpleBufferBind<R, V> {
    type Target = wgpu::BindGroup;

    fn deref(&self) -> &Self::Target {
        &self.bind
    }
}

/// The layout for [`SimpleBufferBind`]
pub struct SimpleBufferBindLayout<V: Visibility>(wgpu::BindGroupLayout, PhantomData<V>);

impl<V: Visibility> SimpleBufferBindLayout<V> {
    /// Create a new layout
    pub fn new(ctx: &GpuContext, label: Option<&str>) -> Self {
        let layout = ctx
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: V::visibility(),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        Self(layout, PhantomData)
    }
}

impl<V: Visibility> std::ops::Deref for SimpleBufferBindLayout<V> {
    type Target = wgpu::BindGroupLayout;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Defines the set of visibility markers
///
/// The ZST's implementing this describe on which stage a buffer is visible in the shader.
pub trait Visibility {
    /// Get the stage of this visibility
    fn visibility() -> wgpu::ShaderStages;
}

/// Visibility marker for just the vertex stage
pub struct VisibleVertex;
/// Visibility marker for just the fragment stage
pub struct VisibleFragment;
/// Visibility marker for both the vertex and fragment stage
#[expect(unused)]
pub struct VisibleBoth;

impl Visibility for VisibleVertex {
    fn visibility() -> wgpu::ShaderStages {
        wgpu::ShaderStages::VERTEX
    }
}
impl Visibility for VisibleFragment {
    fn visibility() -> wgpu::ShaderStages {
        wgpu::ShaderStages::FRAGMENT
    }
}
impl Visibility for VisibleBoth {
    fn visibility() -> wgpu::ShaderStages {
        wgpu::ShaderStages::VERTEX_FRAGMENT
    }
}
