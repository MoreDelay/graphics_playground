struct KernelInfo {
    axis: u32, // 0 for x, 1 for y
    offset: u32,
}

@group(0)
@binding(0)
var src: texture_storage_2d<rgba8unorm, read>;
@group(0)
@binding(1)
var dst: texture_storage_2d<rgba8unorm, write>;

@group(1)
@binding(0)
var kernel: texture_storage_1d<r32float, read>;
@group(1)
@binding(1)
var<uniform> kernel_info: KernelInfo;

@compute
@workgroup_size(16, 16, 1)
fn apply_kernel(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let dim = textureDimensions(src);
    let pos = gid.xy;

    if (pos.x >= dim.x || pos.y >= dim.y) {
        return;
    }

    var dim_max: u32;
    var center: u32;
    if kernel_info.axis == 0 {
        dim_max = dim.x - 1;
        center = pos.x;
    } else {
        dim_max = dim.y - 1;
        center = pos.y;
    }

    let n_kernel = textureDimensions(kernel);
    var value: vec4<f32> = vec4(0, 0, 0, 0);
    for (var i: u32 = 0; i < n_kernel; i++) {
        // clamp-to-edge sampling
        var axis_pos: u32;
        if center + i < kernel_info.offset {
            axis_pos = 0;
        } else {
            axis_pos = min(dim_max, center + i - kernel_info.offset);
        }

        var sample_pos: vec2<u32>;
        if kernel_info.axis == 0 {
            sample_pos = vec2(axis_pos, pos.y);
        } else {
            sample_pos = vec2(pos.x, axis_pos);
        }

        let t = textureLoad(src, sample_pos);
        let k = textureLoad(kernel, i).x;

        value += k * t;
    }

    textureStore(dst, pos, value);
}

