//! Pipeline instruments

pub mod filter;
pub mod halfing;
pub mod image;
pub mod passthru;
pub mod physics;

/// The available filters to use with the image viewer widget
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFilter {
    /// The "nearest" filter
    #[default]
    Nearest,
    /// The "bilinear" filter
    BiLinear,
    /// The "lanczos" filter
    Lanczos,
}
