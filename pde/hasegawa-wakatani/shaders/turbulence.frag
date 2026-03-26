#version 450

in vec2 uv;         // Interpolated uv-coordinates recieved from the Vertex shader
out vec4 fragColor; // Fragment RGBA color to be written to the screen
uniform sampler2D field_texture;   // More like color from the LUT

void main()
{
    fragColor = texture(field_texture, uv);
}