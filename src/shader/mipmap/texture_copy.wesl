import package::vertex::VSOutput;

@group(0)
@binding(0)
var t_src: texture_2d<f32>;
@group(0) @binding(1)
var s_src: sampler;

@fragment
fn fs_copy(vs_in: VSOutput) -> @location(0) vec4<f32> {
    return textureSample(t_src, s_src, vs_in.uv);
}

