import package::image::target::ImageInfo;

fn bilinear_weights(fract: vec2<f32>) -> mat2x2<f32> {
    let tl = (1. - fract.x) * (1. - fract.y);
    let bl = (1. - fract.x) * fract.y;
    let left = vec2(tl, bl);

    let tr = fract.x * (1. - fract.y);
    let br = fract.x * fract.y;
    let right = vec2(tr, br);

    return mat2x2(left, right);
}

fn filter_bilinear(
    image_pos: vec2<f32>,
    t_image: texture_2d<f32>,
    zoom: f32,
) -> vec4<f32> {
    let dims = textureDimensions(t_image);
    let dims_f = vec2<f32>(dims);
    let inside = 0. <= image_pos.x && image_pos.x <= dims_f.x
              && 0. <= image_pos.y && image_pos.y <= dims_f.y;
    if !inside { discard; }

    let level_f = max(0., floor(log2(1. / zoom)));
    let level = u32(level_f);

    let pos_level_f = image_pos / exp2(level_f);
    let dims_level = dims >> vec2(level, level);

    let shift_lo = pos_level_f - vec2(0.5, 0.5);
    let shift_hi = shift_lo + vec2(1., 1.);
    let image_max = vec2<f32>(dims_level - vec2(1, 1));

    let lo = vec2<u32>(clamp(shift_lo, vec2(0., 0.), image_max));
    let hi = vec2<u32>(clamp(shift_hi, vec2(0., 0.), image_max));

    let lo_lo = vec2(lo.x, lo.y);
    let lo_hi = vec2(lo.x, hi.y);
    let hi_lo = vec2(hi.x, lo.y);
    let hi_hi = vec2(hi.x, hi.y);

    let weights = bilinear_weights(fract(shift_lo));
    let lerp = weights[0][0] * textureLoad(t_image, lo_lo, level)
             + weights[0][1] * textureLoad(t_image, lo_hi, level)
             + weights[1][0] * textureLoad(t_image, hi_lo, level)
             + weights[1][1] * textureLoad(t_image, hi_hi, level);
    return lerp;
}
