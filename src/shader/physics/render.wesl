struct VertexIn {
    @location(0) pos: vec2<f32>,
    @location(1) color: vec3<f32>,
}

struct Instance {
    @location(2) model0: vec3<f32>,
    @location(3) model1: vec3<f32>,
    @location(4) model2: vec3<f32>,
}

struct CameraInfo {
    start: vec2<f32>,
    size: vec2<f32>,
}

struct VertexOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) color: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera_info: CameraInfo;

@vertex
fn vs_physics(model: VertexIn, instance: Instance) -> VertexOut {
    var out: VertexOut;
    let model = mat3x3<f32>(
        instance.model0,
        instance.model1,
        instance.model2,
    );

    let world_position = model_matrix * vec3<f32>(model.position, 1.0);
    let camera_uv = (world_position.xy - camera_info.start) / camera_info.size;
    out.clip_position = 2. * camera_uv - vec2<f32>(1., 1.);
    out.color = model.color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.);
}
