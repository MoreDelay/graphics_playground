#version 450

// define Uniform Buffer Object (UBO) that gets updated every frame
// Binding is the index into the descriptor layout.
// Binding multiple descriptor sets simultaneously also works. You then change the set index.
// This way you can access both object-specific and shared descriptor sets without a lot of rebinding.
layout(set = 0, binding = 0) uniform UniformBufferObject {
    vec2 foo; // example on how memory alignment works
    mat4 view;
    mat4 proj;
} ubo;

// Use push constants to provide the model transformation.
layout(push_constant) uniform PushConstants {
    mat4 model;
} pcs;

// define vertex attributes
// This uses the layout defined with the Vertex Input Description.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec2 inTexCoord;
// Some data types use 2 slots, such as 64-bit vertices dvec3. Then you need to skip one index.

// output variable defined by us
// location refers to the framebuffer
layout(location = 0) out vec3 fragColor;
layout(location = 1) out vec2 fragTexCoord;

void main() {
    // predefind variables start with gl_ and are set specifically for the current vertex
    // position is a predefined output variable, and e.g. gl_VertexId is a predefined value.
    gl_Position = ubo.proj * ubo.view * pcs.model * vec4(inPosition, 1.0);

    fragColor = inColor;
    fragTexCoord = inTexCoord;
}
