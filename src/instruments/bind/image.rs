//! Bind groups related to image rendering

/// Viewport transform matrix
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ViewportRaw {
    /// vector 0 of column-major homogeneous transformation matrix
    pub view0: [f32; 3],
    /// padding 0
    pub _pad0: u32,
    /// vector 1 of column-major homogeneous transformation matrix
    pub view1: [f32; 3],
    /// padding 1
    pub _pad1: u32,
    /// vector 2 of column-major homogeneous transformation matrix
    pub view2: [f32; 3],
    /// padding 2
    pub _pad2: u32,
}

/// Raw lanczos metadata for shaders
#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LanczosInfoRaw {
    /// Size of windowing function, typically integer values and either 2 or 3
    ///
    /// This also determines the number of lobes included, where we have a total of $2 a - 1$ lobes
    /// with $a$ being the filter size.
    pub filter_size: f32,
}
