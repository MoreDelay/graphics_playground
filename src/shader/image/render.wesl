import package::image::target::ImageInfo;
import package::image::filter::bilinear::filter_bilinear;
import package::image::filter::nearest::filter_nearest;
import package::image::filter::lanczos::{filter_lanczos, LanczosInfo};

@group(0) @binding(0)
var t_image: texture_2d<f32>;

@group(1) @binding(0)
var<uniform> image_info: ImageInfo;

@if(FILTER_LANCZOS)
@group(2)
@binding(0)
var<uniform> lanczos_info: LanczosInfo;

@fragment
fn fs_image(@builtin(position) in: vec4<f32>) -> @location(0) vec4<f32> {
    @if(FILTER_BILINEAR)
    let color = filter_bilinear(t_image, image_info, in);
    @elif(FILTER_LANCZOS)
    let color = filter_lanczos(t_image, image_info, lanczos_info, in);
    @elif(FILTER_NEAREST)
    let color = filter_nearest(t_image, image_info, in);

    return color;
}
