#version 450

layout(set = 0, binding = 0) uniform sampler2D inputColor1;
layout(set = 1, binding = 0) uniform sampler2D inputColor2;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main()
{
    vec3 color1 = texture(inputColor1, fragTexCoord).rgb;
    vec3 color2 = texture(inputColor2, fragTexCoord).rgb;
    
    outColor = vec4(color1 + color2, 1.0f);
}