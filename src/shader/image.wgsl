struct VSOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

struct ImageMetadata {
    size: vec2<f32>,
    start: vec2<f32>,
    area: vec2<f32>,
}

@group(0) @binding(0)
var t_image: texture_2d<f32>;
@group(0) @binding(1)
var s_image: sampler;

@group(1) @binding(0)
var<uniform> metadata: ImageMetadata;

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

@fragment
fn fs_main(in: VSOutput) -> @location(0) vec4<f32> {
    let uv = in.uv * 1.5 - 0.25;

    let inside = 0. <= uv.x && uv.x <= 1. && 0. <= uv.y && uv.y <= 1. ;
    if !inside { discard; }
    let color = textureSample(t_image, s_image, uv);
    return color;
}
