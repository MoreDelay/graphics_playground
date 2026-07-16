import package::vertex::VSOutput;

struct ViewInfo {
    view_size: vec2<f32>,
    start: vec2<f32>,
    scale: f32,
}

@group(0) @binding(0)
var t_image: texture_2d<f32>;
@group(0) @binding(1)
var s_image: sampler;

@group(1) @binding(0)
var<uniform> view_info: ViewInfo;

fn bilinear_weights(fract: vec2<f32>) -> mat2x2<f32> {
    let tl = (1. - fract.x) * (1. - fract.y);
    let bl = (1. - fract.x) * fract.y;
    let left = vec2(tl, bl);

    let tr = fract.x * (1. - fract.y);
    let br = fract.x * fract.y;
    let right = vec2(tr, br);

    return mat2x2(left, right);
}

// Use integer coords as otherwise some pixels are jumping on large images due to floating point errors.
fn sample_at(coords: vec2<i32>, level: u32) -> vec4<f32> {
    let dims = vec2<i32>(textureDimensions(t_image));

    let inside = (0 <= coords.x && coords.x < dims.x)
              && (0 <= coords.y && coords.y < dims.y);
    if !inside { discard; }

    let sample = textureLoad(t_image, coords, level);
    return sample;
}

fn filter_bilinear(pos_corner: vec2<f32>) -> vec4<f32> {
    // Lerp assumes a sampled light is at a infinitesimal point.
    // Move this point to the center of the pixel.
    let pos_center = pos_corner - vec2(0.5, 0.5);

    let level = u32(max(0., floor(log2(1. / view_info.scale))));
    let pos = pos_center / exp2(f32(level));

    let weights = bilinear_weights(fract(pos));
    let grid_pos = vec2<i32>(floor(pos));
    let lerp = weights[0][0] * sample_at(grid_pos + vec2(0, 0), level)
             + weights[0][1] * sample_at(grid_pos + vec2(0, 1), level)
             + weights[1][0] * sample_at(grid_pos + vec2(1, 0), level)
             + weights[1][1] * sample_at(grid_pos + vec2(1, 1), level);
    return lerp;
}

fn filter_nearest(pos: vec2<f32>) -> vec4<f32> {
    return sample_at(vec2<i32>(floor(pos)), 0);
}

@fragment
fn fs_image(in: VSOutput) -> @location(0) vec4<f32> {
    let viewport_uv = in.uv;
    let viewport_pos = viewport_uv * view_info.view_size;
    let image_pos = (viewport_pos - view_info.start) / view_info.scale;

    @if(FILTER_BILINEAR)
    let color = filter_bilinear(image_pos);
    @else
    let color = filter_nearest(image_pos);

    return color;
}
