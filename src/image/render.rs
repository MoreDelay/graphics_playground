pub mod lanczos;
pub mod mipmap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ImageFilter {
    #[default]
    Nearest,
    BiLinear,
    Lanczos,
}
