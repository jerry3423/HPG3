#version 450

layout(set = 0, binding = 0) uniform sampler2D inputColor;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

const int radius = 22;
const float sigma = 9.0;

float calculate_weights(float x)
{
    return (1.0 / (sqrt(2.0 * 3.14159265 * sigma * sigma))) * exp(-(x * x) / (2.0 * sigma * sigma));
}

void main()
{
    vec3 pixelColor = texture(inputColor, fragTexCoord).rgb;

	vec2 tex_offset = 1.0 / textureSize(inputColor, 0);
	vec3 result = texture(inputColor, fragTexCoord).rgb * calculate_weights(0.0f);

	for(int i = 1; i < 23; i++)
	{
		float weight = calculate_weights(float(i));
		result += texture(inputColor, fragTexCoord + vec2(tex_offset.x * i, 0.0)).rgb * weight;
		result += texture(inputColor, fragTexCoord - vec2(tex_offset.x * i, 0.0)).rgb * weight;
	}

	outColor = vec4(result, 1.0f);
	//outColor = vec4(texture(inputColor, fragTexCoord).rgb, 1.0f);
}