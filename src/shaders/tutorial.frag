#version 450

// define an input variable that also has to be set by the vertex shader
// not the name but the framebuffer location is important
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

// We can reference descriptors in the fragment shader just like we do in the vertex shader.
// A texture sampler is accesses with the sampler2D uniform.
layout(binding = 1) uniform sampler2D texSampler;

// Get push constant for opacity.
layout(push_constant) uniform PushConstants {
    // The first push constant is the model transform.
    layout(offset = 64) float opacity;
} pcs;

// there are no predefined output variables for fragment shaders
layout(location = 0) out vec4 outColor;

void main() {
    // outColor = vec4(fragColor, 1.0);

    // Debug texture coordinates by using them as colors.
    // outColor = vec4(fragTexCoord, 0.0, 1.0);

    // Use the texture sampler for color output using the built-in texture function.
    // outColor = texture(texSampler, fragTexCoord);

    // Use a push constant to control opacity.
    outColor = vec4(texture(texSampler, fragTexCoord).rgb, pcs.opacity);
}
