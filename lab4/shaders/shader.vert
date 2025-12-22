#version 450

layout (location = 0) in vec3 v_position;
layout (location = 1) in vec3 v_normal;
layout (location = 2) in vec2 v_uv;

layout (location = 0) out vec3 f_position;
layout (location = 1) out vec3 f_normal;
layout (location = 2) out vec2 f_uv;

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

void main() {
	vec4 position = model * vec4(v_position, 1.0f);
	mat3 normal_matrix = mat3(transpose(inverse(model)));
	vec3 normal = normalize(normal_matrix * v_normal);

	gl_Position = view_projection * position;

	f_position = position.xyz;
	f_normal = normal;
	f_uv = v_uv;
}
