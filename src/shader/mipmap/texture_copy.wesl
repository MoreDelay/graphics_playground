import super::vertex;

@group(0)
@binding(0)
var texture: texture_2d<f32>;
@group(0) @binding(1)
var sampler: sampler;

@fragment
fn fs_copy(vs_in: VSOutput) -> @location(0) vec4<f32> {
    let dim = textureDimensions(src);
    let pos = gid.xy;

    if (pos.x >= dim.x || pos.y >= dim.y) {
        return;
    }

    let value = textureLoad(src, pos);
    textureStore(dst, pos, value);
}

