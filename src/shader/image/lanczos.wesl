struct ImageInfo {
    new_size: vec2<u32>,
}

struct LanczosInfo {
    filter_size: f32,
}


@group(0)
@binding(0)
var<uniform> lanczos_info: LanczosInfo;
@group(0)
@binding(1)
var<uniform> image_info: ImageInfo;

@group(1)
@binding(0)
var src: texture_2d<f32>;

fn sinc(x: f32) -> f32 {
    if x == 0. {
        return 1.;
    }
    return sin(x) / x;
}

fn lanczos_weight(x: f32, a: f32) -> f32 {
    if abs(x) >= a {
        return 0.;
    }
    return sinc(x) * sinc(x / a);
}

@fragment
fn fs_lanczos(@builtin(position) pos_new: vec4<f32>) -> @location(0) vec4<f32> {
    let old_size = vec2<f32>(textureDimensions(src));
    let new_size = vec2<f32>(image_info.new_size);

    // our image samples will be placed at integer values
    let pos_f = (pos_new.xy - vec2(0.5, 0.5)) * (old_size / new_size);
    let pos_u = vec2<u32>(pos_f);

    let a_f = lanczos_info.filter_size;
    let a_u = u32(a_f);
    let n_samples = a_u * 2;

    var value: vec4<f32> = vec4(0., 0., 0., 0.);

    for (var i: u32 = 0; i < n_samples; i++) {
        let sample_x = pos_u.x - a_u + 1 + i;

        let x = pos_f.x - f32(sample_x);
        let weight_x = lanczos_weight(x, a_f);

        for (var j: u32 = 0; j < n_samples; j++) {
            let sample_y = pos_u.y - a_u + 1 + j;

            let y = pos_f.y - f32(sample_y);
            let weight_y = lanczos_weight(y, a_f);

            let sample_pos = vec2(sample_x, sample_y);
            let sample = textureLoad(src, sample_pos, 0);

            let weight = weight_x * weight_y;

            value += weight * sample;
        }
    }

    return value;
}


