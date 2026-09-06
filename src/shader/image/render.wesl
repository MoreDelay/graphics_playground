import package::image::filter::bilinear::filter_bilinear;
import package::image::filter::lanczos::{filter_lanczos, LanczosInfo};
import package::image::filter::nearest::filter_nearest;
import package::vertex::VertexOut;

@group(1) @binding(0)
var t_image: texture_2d<f32>;

@if(FILTER_LANCZOS)
@group(2)
@binding(0)
var<uniform> lanczos_info: LanczosInfo;

@fragment
fn fs_image(in: VertexOut) -> @location(0) vec4<f32> {
    let dims = vec2<f32>(textureDimensions(t_image));

    let in_pos = in.pos + dims / vec2(2., 2.);
    let image_pos = vec2(in_pos.x, dims.y - in_pos.y);

    let zoom = 1. / dpdx(image_pos.x);

    @if(FILTER_NEAREST)
    let color = filter_nearest(image_pos, t_image);
    @elif(FILTER_BILINEAR)
    let color = filter_bilinear(image_pos, t_image, zoom);
    @elif(FILTER_LANCZOS)
    let color = filter_lanczos(image_pos, t_image, lanczos_info);

    return color;
}
