#version 450

layout (location = 0) in vec3 f_position;
layout (location = 1) in vec3 f_normal;
layout (location = 2) in vec2 f_uv;

layout (location = 0) out vec4 final_color;

layout (binding = 0, std140) uniform SceneUniforms {
	mat4 view_projection;
	mat4 light_view_projection;
	vec4 camera_position;
	vec4 ambient_color_intensity;
	vec4 directional_direction_intensity;
	vec4 directional_color;
	vec4 light_counts;
	vec4 shadow_params;
};

layout (binding = 1, std140) uniform ModelUniforms {
	mat4 model;
	vec4 albedo_color;
	vec4 specular_color_shininess;
};

struct PointLight {
	vec4 position_intensity;
	vec4 color;
};

struct SpotLight {
	vec4 position_intensity;
	vec4 direction_inner;
	vec4 color_outer;
};

layout (binding = 2, std430) readonly buffer PointLightBuffer {
	PointLight point_lights[];
};

layout (binding = 3, std430) readonly buffer SpotLightBuffer {
	SpotLight spot_lights[];
};

layout (binding = 4) uniform sampler2D shadow_map;

float shadowFactor(vec3 world_pos, vec3 normal, vec3 light_dir) {
	vec4 light_clip = light_view_projection * vec4(world_pos, 1.0f);
	if (light_clip.w <= 0.0f) {
		return 0.0f;
	}

	vec3 proj = light_clip.xyz / light_clip.w;
	vec2 uv = proj.xy * 0.5f + 0.5f;

	if (uv.x < 0.0f || uv.x > 1.0f || uv.y < 0.0f || uv.y > 1.0f) {
		return 0.0f;
	}

	float current_depth = proj.z;
	if (current_depth < 0.0f || current_depth > 1.0f) {
		return 0.0f;
	}

	float bias = max(shadow_params.x * (1.0f - max(dot(normal, light_dir), 0.0f)),
	                 shadow_params.x * 0.25f);
	float closest = texture(shadow_map, uv).r;
	return (current_depth - bias) > closest ? shadow_params.y : 0.0f;
}

void main() {
	vec3 albedo = albedo_color.rgb;
	vec3 specular_color = specular_color_shininess.rgb;
	float shininess = max(specular_color_shininess.w, 1.0f);
	vec3 normal = normalize(f_normal);
	vec3 view_dir = normalize(camera_position.xyz - f_position);

	vec3 color = albedo * ambient_color_intensity.rgb * ambient_color_intensity.w;

	vec3 dir_light_dir = normalize(-directional_direction_intensity.xyz);
	float dir_shadow = shadowFactor(f_position, normal, dir_light_dir);
	float dir_shadow_scale = 1.0f - dir_shadow;
	float dir_diffuse = max(dot(normal, dir_light_dir), 0.0f);
	vec3 dir_half = normalize(dir_light_dir + view_dir);
	float dir_specular = dir_diffuse > 0.0f
		? pow(max(dot(normal, dir_half), 0.0f), shininess)
		: 0.0f;
	color += (albedo * dir_diffuse + specular_color * dir_specular) * dir_shadow_scale
		* directional_color.rgb * directional_direction_intensity.w;

	int point_count = int(light_counts.x + 0.5f);
	for (int i = 0; i < point_count; ++i) {
		vec3 light_pos = point_lights[i].position_intensity.xyz;
		float intensity = point_lights[i].position_intensity.w;
		vec3 light_color = point_lights[i].color.rgb;

		vec3 light_vec = light_pos - f_position;
		float dist_sq = max(dot(light_vec, light_vec), 0.0001f);
		float dist = sqrt(dist_sq);
		vec3 light_dir = light_vec / dist;
		float attenuation = intensity / dist_sq;

		float diff = max(dot(normal, light_dir), 0.0f);
		vec3 half_dir = normalize(light_dir + view_dir);
		float spec = diff > 0.0f
			? pow(max(dot(normal, half_dir), 0.0f), shininess)
			: 0.0f;

		color += (albedo * diff + specular_color * spec) * light_color * attenuation;
	}

	int spot_count = int(light_counts.y + 0.5f);
	for (int i = 0; i < spot_count; ++i) {
		vec3 light_pos = spot_lights[i].position_intensity.xyz;
		float intensity = spot_lights[i].position_intensity.w;
		vec3 light_color = spot_lights[i].color_outer.rgb;

		vec3 to_frag = f_position - light_pos;
		float dist_sq = max(dot(to_frag, to_frag), 0.0001f);
		float dist = sqrt(dist_sq);
		vec3 light_dir = to_frag / dist;
		vec3 spot_dir = normalize(spot_lights[i].direction_inner.xyz);

		float theta = dot(light_dir, spot_dir);
		float inner_cos = spot_lights[i].direction_inner.w;
		float outer_cos = spot_lights[i].color_outer.w;
		float edge = clamp((theta - outer_cos) / max(inner_cos - outer_cos, 0.0001f), 0.0f, 1.0f);
		float attenuation = intensity / dist_sq * edge;

		vec3 surface_to_light = -light_dir;
		float diff = max(dot(normal, surface_to_light), 0.0f);
		vec3 half_dir = normalize(surface_to_light + view_dir);
		float spec = diff > 0.0f
			? pow(max(dot(normal, half_dir), 0.0f), shininess)
			: 0.0f;

		color += (albedo * diff + specular_color * spec) * light_color * attenuation;
	}

	final_color = vec4(color, 1.0f);
}
