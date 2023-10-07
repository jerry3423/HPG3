#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec2 inTexCoord;
layout(location = 2) in vec3 inNormal;

layout(set = 0, binding = 0) uniform UScene
{
	mat4 camera;
	mat4 projection;
	mat4 projcam;
	vec2 windowSize;
	vec3 lightPosition;
	vec3 lightColor;
} uScene;

layout (location = 0) out vec2 fragTexCoord;
layout (location = 1) out vec3 fragNormal;
layout (location = 2) out vec3 fragPosition;

void main() {
    gl_Position = uScene.projcam * vec4(inPosition, 1.0);
    fragTexCoord = inTexCoord;
    fragNormal = inNormal;
    fragPosition = inPosition;
}