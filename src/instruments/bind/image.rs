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
