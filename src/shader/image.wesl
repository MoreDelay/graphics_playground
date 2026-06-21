struct VSOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

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

@vertex
fn vs_main(@builtin(vertex_index) in_vertex_index: u32) -> VSOutput {
    // generate quad as triangle strip of 4 vertices
    var pos: vec4<f32> = vec4<f32>(-1., -1., 0., 1.);
    var uv: vec2<f32> = vec2<f32>(0., 1.);
    if in_vertex_index == 1 {
        pos.x = 1.;
        uv.x = 1.;
    }
    else if in_vertex_index == 2 {
        pos.y = 1.;
        uv.y = 0.;
    }
    else if in_vertex_index == 3 {
        pos.x = 1.;
        pos.y = 1.;
        uv.x = 1.;
        uv.y = 0.;
    }

    var out: VSOutput;
    out.position = pos;
    out.uv = uv;
    return out;
}

fn bilinear_weights(fract: vec2<f32>) -> mat2x2<f32> {
    let tl = (1. - fract.x) * (1. - fract.y);
    let bl = (1. - fract.x) * fract.y;
    let left = vec2(tl, bl);

    let tr = fract.x * (1. - fract.y);
    let br = fract.x * fract.y;
    let right = vec2(tr, br);

    return mat2x2(left, right);
}

fn sample_at(pos: vec2<f32>) -> vec4<f32> {
    let dims = vec2<f32>(textureDimensions(t_image));

    let center = pos + vec2(0.5, 0.5);
    let uv = center / dims;

    let inside = (0. <= uv.x && uv.x <= 1.)
              && (0. <= uv.y && uv.y <= 1.);
    if !inside { discard; }

    let sample = textureSample(t_image, s_image, uv);
    return sample;
}

fn filter_bilinear(pos: vec2<f32>) -> vec4<f32> {
    let weights = bilinear_weights(fract(pos));
    let grid_pos = floor(pos);
    let lerp = weights[0][0] * sample_at(grid_pos + vec2(0., 0.))
             + weights[0][1] * sample_at(grid_pos + vec2(0., 1.))
             + weights[1][0] * sample_at(grid_pos + vec2(1., 0.))
             + weights[1][1] * sample_at(grid_pos + vec2(1., 1.));
    return lerp;
}

fn filter_nearest(pos: vec2<f32>) -> vec4<f32> {
    return sample_at(pos);
}

@fragment
fn fs_main(in: VSOutput) -> @location(0) vec4<f32> {
    let viewport_uv = in.uv;
    let viewport_pos = viewport_uv * view_info.view_size;
    let image_pos = (viewport_pos - view_info.start) / view_info.scale;

    @if(FILTER_BILINEAR)
    let color = filter_bilinear(image_pos);
    @else
    let color = filter_nearest(image_pos);

    return color;
}
