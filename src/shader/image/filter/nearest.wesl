import package::image::target::ImageInfo;

fn filter_nearest(
    t_image: texture_2d<f32>,
    image_info: ImageInfo,
    in: vec4<f32>,
) -> vec4<f32> {
    let viewport_pos = in.xy;
    let image_pos = (viewport_pos - image_info.start) / image_info.zoom;

    let dims = vec2<f32>(textureDimensions(t_image));

    let inside = 0 <= image_pos.x && image_pos.x < dims.x
              && 0 <= image_pos.y && image_pos.y < dims.y;
    if !inside { discard; }

    let coords = vec2<u32>(floor(image_pos));
    return textureLoad(t_image, coords, 0);
}
