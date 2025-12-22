#version 450

layout (location = 0) in vec3 v_position;

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

void main() {
	gl_Position = light_view_projection * model * vec4(v_position, 1.0f);
}
