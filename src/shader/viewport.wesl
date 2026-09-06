import package::vertex::{Instance, VertexIn, VertexOut};

struct ViewportInfo {
    // View matrix
    view: mat3x3<f32>,
}

@group(0) @binding(0)
var<uniform> viewport: ViewportInfo;

@vertex
fn vs_viewport(vertex: VertexIn, instance: Instance) -> VertexOut {
    let model = mat3x3(
        instance.model0,
        instance.model1,
        instance.model2,
    );

    let pos = model * vec3(vertex.pos, 1.0);
    let clip = viewport.view * pos;

    var out: VertexOut;
    out.clip = vec4(clip.xy, 0., clip.z);
    out.pos = pos.xy / pos.z;
    out.color = vertex.color;
    out.uv = vertex.uv;
    return out;
}
