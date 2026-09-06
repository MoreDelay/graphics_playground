import package::image::target::ImageInfo;

struct LanczosInfo {
    filter_size: f32,
}

const PI = radians(180.0);

fn sinc(x: f32) -> f32 {
    if x == 0. {
        return 1.;
    }
    return sin(PI * x) / (PI * x);
}

fn lanczos_weight(x: f32, a: f32) -> f32 {
    if abs(x) >= a {
        return 0.;
    }
    return sinc(x) * sinc(x / a);
}

fn filter_lanczos(
    image_pos: vec2<f32>,
    src: texture_2d<f32>,
    lanczos_info: LanczosInfo,
) -> vec4<f32> {
    let image_dims = textureDimensions(src);

    let inside = 0. <= image_pos.x && image_pos.x <= f32(image_dims.x)
              && 0. <= image_pos.y && image_pos.y <= f32(image_dims.y);
    if !inside {
        discard;
    }

    let image_max = image_dims - vec2(1, 1);

    let pos_f = image_pos - vec2(0.5,0.5);
    let pos_u = vec2<u32>(pos_f);

    let a_f = lanczos_info.filter_size;
    let a_u = u32(a_f);
    let n_samples = a_u * 2; // `a_u` samples left and right from `pos_f`

    var value: vec4<f32> = vec4(0., 0., 0., 0.);
    var weights: f32 = 0.;

    for (var i: u32 = 0; i < n_samples; i++) {
        var sample_x: u32 = 0;
        if (pos_u.x + 1 + i > a_u) {
            sample_x = pos_u.x - a_u + 1 + i;
        }
        sample_x = min(sample_x, image_max.x);

        let x = pos_f.x - f32(sample_x);
        let weight_x = lanczos_weight(x, a_f);

        for (var j: u32 = 0; j < n_samples; j++) {
            var sample_y: u32 = 0;
            if (pos_u.y + 1 + j > a_u) {
                sample_y = pos_u.y - a_u + 1 + j;
            }
            sample_y = min(sample_y, image_max.y);

            let y = pos_f.y - f32(sample_y);
            let weight_y = lanczos_weight(y, a_f);
            let weight = weight_x * weight_y;

            let sample_pos = vec2(sample_x, sample_y);
            let sample = textureLoad(src, sample_pos, 0);

            value += weight * sample;
            weights += weight;
        }
    }

    return value / weights;
}
