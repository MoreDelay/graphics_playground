@group(0)
@binding(0)
var t_src: texture_2d<f32>;

@fragment
fn fs_copy(@builtin(position) vs_in: vec4<f32>) -> @location(0) vec4<f32> {
    let pos = vec2<u32>(vs_in.xy);
    return textureLoad(t_src, pos, 0);
}

