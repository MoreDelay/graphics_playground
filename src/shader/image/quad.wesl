import package::vertex::VertexOut;

@vertex
fn vs_quad(@builtin(vertex_index) in_vertex_index: u32) -> VertexOut {
    // generate quad as triangle strip of 4 vertices
    var pos: vec4<f32> = vec4<f32>(-1., -1., 0., 1.);
    var uv: vec2<f32> = vec2<f32>(0., 1.);
    if in_vertex_index == 1 {
        pos.x = 1.;
        uv.x = 1.;
    }
    else if in_vertex_index == 2 {
        pos.y = 1.;
        uv.y = 0.;
    }
    else if in_vertex_index == 3 {
        pos.x = 1.;
        pos.y = 1.;
        uv.x = 1.;
        uv.y = 0.;
    }

    var out: VertexOut;
    out.pos = pos;
    out.uv = uv;
    return out;
}

