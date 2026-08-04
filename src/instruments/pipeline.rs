pub mod filter;
pub mod halfing;
pub mod image;
pub mod passthru;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFilter {
    #[default]
    Nearest,
    BiLinear,
    Lanczos,
}
