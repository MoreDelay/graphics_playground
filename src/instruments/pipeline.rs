//! Pipeline instruments

use std::num::NonZeroU8;

use derive_more::Display;

pub mod filter;
pub mod halfing;
pub mod image;
pub mod passthru;
pub mod physics;

/// Non-zero 2 for defining lobes
pub const TWO: NonZeroU8 = NonZeroU8::new(2).unwrap();
/// Non-zero 3 for defining lobes
pub const THREE: NonZeroU8 = NonZeroU8::new(3).unwrap();

/// The available filters to use with the image viewer widget
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFilter {
    /// The "nearest" filter
    #[default]
    Nearest,
    /// The "bilinear" filter
    BiLinear,
    /// The "lanczos" filter with 2 lobes
    Lanczos(NonZeroU8),
}

impl ImageFilter {
    /// All filter types used actively by the app
    pub const ALL: [Self; 4] = [
        Self::Nearest,
        Self::BiLinear,
        Self::Lanczos(TWO),
        Self::Lanczos(THREE),
    ];

    /// Get the filter type of this image filter
    pub const fn as_type(self) -> ImageFilterType {
        match self {
            Self::Nearest => ImageFilterType::Nearest,
            Self::BiLinear => ImageFilterType::BiLinear,
            Self::Lanczos(_) => ImageFilterType::Lanczos,
        }
    }
}

/// The available filters types without filter parameters
#[derive(Debug, Display, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFilterType {
    /// The "nearest" filter
    #[default]
    Nearest,
    /// The "bilinear" filter
    BiLinear,
    /// The "lanczos" filter
    Lanczos,
}

impl std::fmt::Display for ImageFilter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nearest => f.write_str("Nearest"),
            Self::BiLinear => f.write_str("BiLinear"),
            Self::Lanczos(lobes) => f.write_str(&format!("Lanczos {lobes}")),
        }
    }
}
