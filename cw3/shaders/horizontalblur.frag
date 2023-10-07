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
    //calculate_weights();
    float weights[23] = float[] (0.0281,	0.0281,	0.0279,	0.0277,	0.0274,	0.0270,	
	0.0266,	0.0261,	0.0255,	0.0248,	0.0241,	0.0233,	0.0225,	0.0216,	0.0208,	
	0.0199,	0.0189,	0.0180,	0.0170,	0.0161,	0.0152,	0.0142,	0.0133);
    vec3 pixelColor = texture(inputColor, fragTexCoord).rgb;

	vec2 tex_offset = 1.0 / textureSize(inputColor, 0);
	vec3 result = texture(inputColor, fragTexCoord).rgb * calculate_weights(0.0f);

	for(int i = 1; i < 23; i++)
	{
		float weight = calculate_weights(float(i));
		result += texture(inputColor, fragTexCoord + vec2(0.0, tex_offset.y * i)).rgb * weight;
		result += texture(inputColor, fragTexCoord - vec2(0.0, tex_offset.y * i)).rgb * weight;
	}

	outColor = vec4(result, 1.0f);
	//outColor = vec4(texture(inputColor, fragTexCoord).rgb, 1.0f);
}