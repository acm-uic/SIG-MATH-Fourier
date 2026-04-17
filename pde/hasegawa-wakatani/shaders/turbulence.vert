#version 460

// Input
in vec2 position_in;
in vec2 uv_in;
uniform vec2 offset; // Screen offset:   (0,0) for left panel, (1,0) for right panel

// Outputs
out vec2 uv;    // Coordinates for fragment shader

void main()
{
    // Convert the Normalized Coordinates
    vec2 p = 0.5*position_in + 0.5;

    // Determining the panel location based on the x-coordinates
    p.x = (p.x + offset.x)*0.5;

    // Convert back to Normalized Coordinates
    p = 2.0*p - 1.0;

    // Finalized the position
    gl_Position = vec4(p, 0.0, 1.0);
    uv = uv_in;
}