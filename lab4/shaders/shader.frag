#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;
layout (location = 3) in vec4 f_light_space;

layout (location = 0) out vec4 final_color;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	mat4 light_view_projection;
	vec4 camera_position;
	vec4 ambient_color_intensity;
	vec4 directional_direction_intensity;
	vec4 directional_color;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec4 albedo_color;
	vec4 specular_color_shininess;
};

layout (binding = 2) uniform sampler2D base_texture;
layout (binding = 3) uniform sampler2DShadow shadow_map;

float sampleShadow(vec4 light_space, vec3 normal, vec3 light_dir) {
	vec3 proj = light_space.xyz / light_space.w;
	vec2 uv = proj.xy * 0.5f + 0.5f;

	if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f || proj.z > 1.0f) {
		return 1.0f;
	}

	float bias = max(0.0015f * (1.0f - dot(normal, light_dir)), 0.0005f);
	return texture(shadow_map, vec3(uv, proj.z - bias));
}

void main() {
	vec3 albedo = albedo_color.rgb * texture(base_texture, f_uv).rgb;
	vec3 specular_color = specular_color_shininess.rgb;
	float shininess = max(specular_color_shininess.w, 1.0f);
	vec3 normal = normalize(f_normal);
	vec3 view_dir = normalize(camera_position.xyz - f_position);

	vec3 light_dir = normalize(-directional_direction_intensity.xyz);
	float shadow = sampleShadow(f_light_space, normal, light_dir);

	vec3 color = albedo * ambient_color_intensity.rgb * ambient_color_intensity.w;

	float diff = max(dot(normal, light_dir), 0.0f);
	vec3 half_dir = normalize(light_dir + view_dir);
	float spec = diff > 0.0f
		? pow(max(dot(normal, half_dir), 0.0f), shininess)
		: 0.0f;

	color += shadow * (albedo * diff + specular_color * spec)
		* directional_color.rgb * directional_direction_intensity.w;

	final_color = vec4(color, 1.0f);
}
