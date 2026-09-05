//! Bindings used for physics sim

use iced::wgpu;

/// Raw image metadata for shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraInfoRaw {
    /// (x, y) of bottom left corner of visible area
    pub start: [f32; 2],
    /// (width, height) of the visible area
    pub size: [f32; 2],
}

/// The format expected by shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexRaw {
    /// The position coordinates of this vertex
    pub pos: [f32; 2],
    /// Padding to satisfy wgpu alignment constraints
    pub _pad1: [u32; 2],
    /// The color of this vertex
    pub color: [f32; 3],
    /// Padding to satisfy wgpu alignment constraints
    pub _pad2: u32,
}

impl VertexRaw {
    /// Create a new raw vertex
    ///
    /// Just a helper to eliminate noise about the padding
    pub const fn new(pos: [f32; 2], color: [f32; 3]) -> Self {
        Self {
            pos,
            _pad1: [0, 0],
            color,
            _pad2: 0,
        }
    }

    /// Create the vertex descriptor for these instances
    pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

/// The raw data layout for an instance, passed as vertex attributes
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct InstanceRaw {
    /// column 0 of homogeneous transformation matrix in world space
    pub model0: [f32; 3],
    /// Padding 0
    pub _pad0: u32,
    /// column 1 of homogeneous transformation matrix in world space
    pub model1: [f32; 3],
    /// Padding 1
    pub _pad1: u32,
    /// column 2 of homogeneous transformation matrix in world space
    pub model2: [f32; 3],
    /// Padding 2
    pub _pad2: u32,
}

impl InstanceRaw {
    /// Create a new raw instance
    ///
    /// Just a helper to eliminate noise about the padding. Vectors are column vectors of the
    /// homogeneous transformation matrix.
    pub const fn new(model0: [f32; 3], model1: [f32; 3], model2: [f32; 3]) -> Self {
        Self {
            model0,
            _pad0: 0,
            model1,
            _pad1: 0,
            model2,
            _pad2: 0,
        }
    }

    /// Create the vertex descriptor for these instances
    pub const fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}
