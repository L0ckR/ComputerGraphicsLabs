#include <cstdint>
#include <climits>
#include <cstring>
#include <cstdio>
#include <array>
#include <vector>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>

#include <veekay/veekay.hpp>

#include <vulkan/vulkan_core.h>
#include <imgui.h>
#include <lodepng.h>

namespace {

constexpr uint32_t max_models = 1024;
constexpr uint32_t max_point_lights = 8;
constexpr uint32_t max_spot_lights = 8;

struct Vertex {
	veekay::vec3 position;
	veekay::vec3 normal;
	veekay::vec2 uv;
	// NOTE: You can add more attributes
};

struct SceneUniforms {
	veekay::mat4 view_projection;
	veekay::vec4 camera_position;
	veekay::vec4 ambient_color_intensity;
	veekay::vec4 directional_direction_intensity;
	veekay::vec4 directional_color;
	veekay::vec4 light_counts;
};

struct ModelUniforms {
	veekay::mat4 model;
	veekay::vec4 albedo_color;
	veekay::vec4 specular_color_shininess;
};

struct AmbientLightSettings {
	veekay::vec3 color = {0.08f, 0.08f, 0.1f};
	float intensity = 0.4f;
};

struct DirectionalLightSettings {
	veekay::vec3 direction = {0.3f, -1.0f, 0.2f};
	float intensity = 1.0f;
	veekay::vec3 color = {1.0f, 1.0f, 1.0f};
	float _pad0 = 0.0f;
};

struct PointLightSettings {
	veekay::vec3 position = {};
	float intensity = 1.0f;
	veekay::vec3 color = {1.0f, 1.0f, 1.0f};
	float _pad0 = 0.0f;
};

struct SpotLightSettings {
	veekay::vec3 position = {};
	float intensity = 1.0f;
	veekay::vec3 direction = {0.0f, 0.0f, 1.0f};
	float inner_angle_degrees = 15.0f;
	veekay::vec3 color = {1.0f, 1.0f, 1.0f};
	float outer_angle_degrees = 25.0f;
};

struct PointLight {
	veekay::vec4 position_intensity;
	veekay::vec4 color;
};

struct SpotLight {
	veekay::vec4 position_intensity;
	veekay::vec4 direction_inner;
	veekay::vec4 color_outer;
};

struct Mesh {
	veekay::graphics::Buffer* vertex_buffer;
	veekay::graphics::Buffer* index_buffer;
	uint32_t indices;
};

struct Transform {
	veekay::vec3 position = {};
	veekay::vec3 scale = {1.0f, 1.0f, 1.0f};
	veekay::vec3 rotation = {};

	// NOTE: Model matrix (translation, rotation and scaling)
	veekay::mat4 matrix() const;
};

struct Model {
	Mesh mesh;
	Transform transform;
	veekay::vec3 albedo_color;
	veekay::vec3 specular_color;
	float shininess = 32.0f;
};

struct CameraTransformState {
	veekay::vec3 position = {};
	veekay::vec3 rotation = {};
};

struct CameraLookAtState {
	veekay::vec3 target = {};
	veekay::vec2 rotation = {};
	float distance = 3.0f;
};

enum class CameraMode {
	transform,
	look_at,
};

struct Camera {
	constexpr static float default_fov = 60.0f;
	constexpr static float default_near_plane = 0.01f;
	constexpr static float default_far_plane = 100.0f;

	veekay::vec3 position = {};
	veekay::vec3 rotation = {};

	float fov = default_fov;
	float near_plane = default_near_plane;
	float far_plane = default_far_plane;
	CameraMode mode = CameraMode::transform;

	veekay::vec2 look_at_rotation = {};
	veekay::vec3 look_at_target = {};
	float look_at_distance = 3.0f;

	// NOTE: View matrix of camera (inverse of a transform)
	veekay::mat4 view() const;

	// NOTE: View and projection composition
	veekay::mat4 view_projection(float aspect_ratio) const;

	veekay::vec3 world_position() const;
};

// NOTE: Scene objects
inline namespace {
	Camera camera{
		.position = {0.0f, -0.6f, -3.5f},
		.mode = CameraMode::look_at,
		.look_at_target = {0.0f, -0.4f, 0.2f},
		.look_at_distance = 3.6f,
	};

	std::vector<Model> models;
	AmbientLightSettings ambient_light;
	DirectionalLightSettings directional_light;
	std::array<PointLightSettings, max_point_lights> point_lights_settings{};
	std::array<SpotLightSettings, max_spot_lights> spot_lights_settings{};
	uint32_t point_light_count = 2;
	uint32_t spot_light_count = 1;
	CameraTransformState saved_transform_state;
	CameraLookAtState saved_look_at_state;
	bool saved_transform_valid = false;
	bool saved_look_at_valid = false;
}

// NOTE: Vulkan objects
inline namespace {
	VkShaderModule vertex_shader_module;
	VkShaderModule fragment_shader_module;

	VkDescriptorPool descriptor_pool;
	VkDescriptorSetLayout descriptor_set_layout;
	VkDescriptorSet descriptor_set;

	VkPipelineLayout pipeline_layout;
	VkPipeline pipeline;

	veekay::graphics::Buffer* scene_uniforms_buffer;
	veekay::graphics::Buffer* model_uniforms_buffer;
	veekay::graphics::Buffer* point_lights_buffer;
	veekay::graphics::Buffer* spot_lights_buffer;

	Mesh plane_mesh;
	Mesh cube_mesh;

	veekay::graphics::Texture* missing_texture;
	VkSampler missing_texture_sampler;

	veekay::graphics::Texture* texture;
	VkSampler texture_sampler;
}

float toRadians(float degrees) {
	return degrees * float(M_PI) / 180.0f;
}

veekay::vec3 directionFromAngles(const veekay::vec2& angles) {
	float pitch = angles.x;
	float yaw = angles.y;
	float cos_pitch = std::cos(pitch);

	return {
		std::sin(yaw) * cos_pitch,
		-std::sin(pitch),
		std::cos(yaw) * cos_pitch,
	};
}

veekay::mat4 lookAtMatrix(const veekay::vec3& position, const veekay::vec3& target) {
	const veekay::vec3 world_up = {0.0f, -1.0f, 0.0f};
	veekay::vec3 front = veekay::vec3::normalized(target - position);
	veekay::vec3 right = veekay::vec3::normalized(veekay::vec3::cross(front, world_up));
	veekay::vec3 up = veekay::vec3::cross(right, front);

	veekay::mat4 view = veekay::mat4::identity();
	view[0][0] = right.x;
	view[0][1] = right.y;
	view[0][2] = right.z;
	view[1][0] = up.x;
	view[1][1] = up.y;
	view[1][2] = up.z;
	view[2][0] = front.x;
	view[2][1] = front.y;
	view[2][2] = front.z;
	view[3][0] = -veekay::vec3::dot(right, position);
	view[3][1] = -veekay::vec3::dot(up, position);
	view[3][2] = -veekay::vec3::dot(front, position);
	view[3][3] = 1.0f;

	return view;
}

CameraLookAtState lookAtFromTransformState(const CameraTransformState& state, float distance_hint) {
	CameraLookAtState result;
	result.rotation = {state.rotation.x, state.rotation.y};
	result.distance = distance_hint > 0.0f ? distance_hint : 3.0f;
	veekay::vec3 front = directionFromAngles(result.rotation);
	result.target = state.position + front * result.distance;

	return result;
}

CameraTransformState transformFromLookAtState(const CameraLookAtState& state) {
	CameraTransformState result;
	veekay::vec3 front = directionFromAngles(state.rotation);
	result.position = state.target - front * state.distance;
	result.rotation = {state.rotation.x, state.rotation.y, 0.0f};

	return result;
}

veekay::mat4 Transform::matrix() const {
	auto s = veekay::mat4::scaling(scale);
	auto rx = veekay::mat4::rotation({1.0f, 0.0f, 0.0f}, rotation.x);
	auto ry = veekay::mat4::rotation({0.0f, 1.0f, 0.0f}, rotation.y);
	auto rz = veekay::mat4::rotation({0.0f, 0.0f, 1.0f}, rotation.z);
	auto t = veekay::mat4::translation(position);

	return t * rz * ry * rx * s;
}

veekay::mat4 Camera::view() const {
	if (mode == CameraMode::look_at) {
		return lookAtMatrix(world_position(), look_at_target);
	}

	veekay::vec3 front = directionFromAngles({rotation.x, rotation.y});
	return lookAtMatrix(position, position + front);
}

veekay::mat4 Camera::view_projection(float aspect_ratio) const {
	auto projection = veekay::mat4::projection(fov, aspect_ratio, near_plane, far_plane);

	return view() * projection;
}

veekay::vec3 Camera::world_position() const {
	if (mode == CameraMode::look_at) {
		veekay::vec3 front = directionFromAngles(look_at_rotation);
		return look_at_target - front * look_at_distance;
	}

	return position;
}

// NOTE: Loads shader byte code from file
// NOTE: Your shaders are compiled via CMake with this code too, look it up
VkShaderModule loadShaderModule(const char* path) {
	std::ifstream file(path, std::ios::binary | std::ios::ate);
	size_t size = file.tellg();
	std::vector<uint32_t> buffer(size / sizeof(uint32_t));
	file.seekg(0);
	file.read(reinterpret_cast<char*>(buffer.data()), size);
	file.close();

	VkShaderModuleCreateInfo info{
		.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
		.codeSize = size,
		.pCode = buffer.data(),
	};

	VkShaderModule result;
	if (vkCreateShaderModule(veekay::app.vk_device, &
	                         info, nullptr, &result) != VK_SUCCESS) {
		return nullptr;
	}

	return result;
}

void initialize(VkCommandBuffer cmd) {
	VkDevice& device = veekay::app.vk_device;
	VkPhysicalDevice& physical_device = veekay::app.vk_physical_device;

	{ // NOTE: Build graphics pipeline
		vertex_shader_module = loadShaderModule("./shaders/shader.vert.spv");
		if (!vertex_shader_module) {
			std::cerr << "Failed to load Vulkan vertex shader from file\n";
			veekay::app.running = false;
			return;
		}

		fragment_shader_module = loadShaderModule("./shaders/shader.frag.spv");
		if (!fragment_shader_module) {
			std::cerr << "Failed to load Vulkan fragment shader from file\n";
			veekay::app.running = false;
			return;
		}

		VkPipelineShaderStageCreateInfo stage_infos[2];

		// NOTE: Vertex shader stage
		stage_infos[0] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_VERTEX_BIT,
			.module = vertex_shader_module,
			.pName = "main",
		};

		// NOTE: Fragment shader stage
		stage_infos[1] = VkPipelineShaderStageCreateInfo{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
			.stage = VK_SHADER_STAGE_FRAGMENT_BIT,
			.module = fragment_shader_module,
			.pName = "main",
		};

		// NOTE: How many bytes does a vertex take?
		VkVertexInputBindingDescription buffer_binding{
			.binding = 0,
			.stride = sizeof(Vertex),
			.inputRate = VK_VERTEX_INPUT_RATE_VERTEX,
		};

		// NOTE: Declare vertex attributes
		VkVertexInputAttributeDescription attributes[] = {
			{
				.location = 0, // NOTE: First attribute
				.binding = 0, // NOTE: First vertex buffer
				.format = VK_FORMAT_R32G32B32_SFLOAT, // NOTE: 3-component vector of floats
				.offset = offsetof(Vertex, position), // NOTE: Offset of "position" field in a Vertex struct
			},
			{
				.location = 1,
				.binding = 0,
				.format = VK_FORMAT_R32G32B32_SFLOAT,
				.offset = offsetof(Vertex, normal),
			},
			{
				.location = 2,
				.binding = 0,
				.format = VK_FORMAT_R32G32_SFLOAT,
				.offset = offsetof(Vertex, uv),
			},
		};

		// NOTE: Describe inputs
		VkPipelineVertexInputStateCreateInfo input_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
			.vertexBindingDescriptionCount = 1,
			.pVertexBindingDescriptions = &buffer_binding,
			.vertexAttributeDescriptionCount = sizeof(attributes) / sizeof(attributes[0]),
			.pVertexAttributeDescriptions = attributes,
		};

		// NOTE: Every three vertices make up a triangle,
		//       so our vertex buffer contains a "list of triangles"
		VkPipelineInputAssemblyStateCreateInfo assembly_state_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
			.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST,
		};

		// NOTE: Declare clockwise triangle order as front-facing
		//       Discard triangles that are facing away
		//       Fill triangles, don't draw lines instaed
		VkPipelineRasterizationStateCreateInfo raster_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
			.polygonMode = VK_POLYGON_MODE_FILL,
			.cullMode = VK_CULL_MODE_BACK_BIT,
			.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE,
			.lineWidth = 1.0f,
		};

		// NOTE: Use 1 sample per pixel
		VkPipelineMultisampleStateCreateInfo sample_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
			.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT,
			.sampleShadingEnable = false,
			.minSampleShading = 1.0f,
		};

		VkViewport viewport{
			.x = 0.0f,
			.y = 0.0f,
			.width = static_cast<float>(veekay::app.window_width),
			.height = static_cast<float>(veekay::app.window_height),
			.minDepth = 0.0f,
			.maxDepth = 1.0f,
		};

		VkRect2D scissor{
			.offset = {0, 0},
			.extent = {veekay::app.window_width, veekay::app.window_height},
		};

		// NOTE: Let rasterizer draw on the entire window
		VkPipelineViewportStateCreateInfo viewport_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,

			.viewportCount = 1,
			.pViewports = &viewport,

			.scissorCount = 1,
			.pScissors = &scissor,
		};

		// NOTE: Let rasterizer perform depth-testing and overwrite depth values on condition pass
		VkPipelineDepthStencilStateCreateInfo depth_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO,
			.depthTestEnable = true,
			.depthWriteEnable = true,
			.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL,
		};

		// NOTE: Let fragment shader write all the color channels
		VkPipelineColorBlendAttachmentState attachment_info{
			.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
			                  VK_COLOR_COMPONENT_G_BIT |
			                  VK_COLOR_COMPONENT_B_BIT |
			                  VK_COLOR_COMPONENT_A_BIT,
		};

		// NOTE: Let rasterizer just copy resulting pixels onto a buffer, don't blend
		VkPipelineColorBlendStateCreateInfo blend_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,

			.logicOpEnable = false,
			.logicOp = VK_LOGIC_OP_COPY,

			.attachmentCount = 1,
			.pAttachments = &attachment_info
		};

		{
			VkDescriptorPoolSize pools[] = {
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 8,
				},
				{
					.type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
					.descriptorCount = 8,
				}
			};
			
			VkDescriptorPoolCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
				.maxSets = 1,
				.poolSizeCount = sizeof(pools) / sizeof(pools[0]),
				.pPoolSizes = pools,
			};

			if (vkCreateDescriptorPool(device, &info, nullptr,
			                           &descriptor_pool) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor pool\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Descriptor set layout specification
		{
			VkDescriptorSetLayoutBinding bindings[] = {
				{
					.binding = 0,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 1,
					.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 2,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
				{
					.binding = 3,
					.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
					.descriptorCount = 1,
					.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT,
				},
			};

			VkDescriptorSetLayoutCreateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
				.bindingCount = sizeof(bindings) / sizeof(bindings[0]),
				.pBindings = bindings,
			};

			if (vkCreateDescriptorSetLayout(device, &info, nullptr,
			                                &descriptor_set_layout) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set layout\n";
				veekay::app.running = false;
				return;
			}
		}

		{
			VkDescriptorSetAllocateInfo info{
				.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
				.descriptorPool = descriptor_pool,
				.descriptorSetCount = 1,
				.pSetLayouts = &descriptor_set_layout,
			};

			if (vkAllocateDescriptorSets(device, &info, &descriptor_set) != VK_SUCCESS) {
				std::cerr << "Failed to create Vulkan descriptor set\n";
				veekay::app.running = false;
				return;
			}
		}

		// NOTE: Declare external data sources, only push constants this time
		VkPipelineLayoutCreateInfo layout_info{
			.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
			.setLayoutCount = 1,
			.pSetLayouts = &descriptor_set_layout,
		};

		// NOTE: Create pipeline layout
		if (vkCreatePipelineLayout(device, &layout_info,
		                           nullptr, &pipeline_layout) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline layout\n";
			veekay::app.running = false;
			return;
		}
		
		VkGraphicsPipelineCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO,
			.stageCount = 2,
			.pStages = stage_infos,
			.pVertexInputState = &input_state_info,
			.pInputAssemblyState = &assembly_state_info,
			.pViewportState = &viewport_info,
			.pRasterizationState = &raster_info,
			.pMultisampleState = &sample_info,
			.pDepthStencilState = &depth_info,
			.pColorBlendState = &blend_info,
			.layout = pipeline_layout,
			.renderPass = veekay::app.vk_render_pass,
		};

		// NOTE: Create graphics pipeline
		if (vkCreateGraphicsPipelines(device, nullptr,
		                              1, &info, nullptr, &pipeline) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan pipeline\n";
			veekay::app.running = false;
			return;
		}
	}

	scene_uniforms_buffer = new veekay::graphics::Buffer(
		sizeof(SceneUniforms),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	model_uniforms_buffer = new veekay::graphics::Buffer(
		max_models * veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms)),
		nullptr,
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);

	point_lights_buffer = new veekay::graphics::Buffer(
		max_point_lights * sizeof(PointLight),
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	spot_lights_buffer = new veekay::graphics::Buffer(
		max_spot_lights * sizeof(SpotLight),
		nullptr,
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);

	// NOTE: This texture and sampler is used when texture could not be loaded
	{
		VkSamplerCreateInfo info{
			.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
			.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE,
		};

		if (vkCreateSampler(device, &info, nullptr, &missing_texture_sampler) != VK_SUCCESS) {
			std::cerr << "Failed to create Vulkan texture sampler\n";
			veekay::app.running = false;
			return;
		}

		uint32_t pixels[] = {
			0xff000000, 0xffff00ff,
			0xffff00ff, 0xff000000,
		};

		missing_texture = new veekay::graphics::Texture(cmd, 2, 2,
		                                                VK_FORMAT_B8G8R8A8_UNORM,
		                                                pixels);
	}

	{
		VkDescriptorBufferInfo buffer_infos[] = {
			{
				.buffer = scene_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(SceneUniforms),
			},
			{
				.buffer = model_uniforms_buffer->buffer,
				.offset = 0,
				.range = sizeof(ModelUniforms),
			},
			{
				.buffer = point_lights_buffer->buffer,
				.offset = 0,
				.range = max_point_lights * sizeof(PointLight),
			},
			{
				.buffer = spot_lights_buffer->buffer,
				.offset = 0,
				.range = max_spot_lights * sizeof(SpotLight),
			},
		};

		VkWriteDescriptorSet write_infos[] = {
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 0,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
				.pBufferInfo = &buffer_infos[0],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 1,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC,
				.pBufferInfo = &buffer_infos[1],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 2,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[2],
			},
			{
				.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
				.dstSet = descriptor_set,
				.dstBinding = 3,
				.dstArrayElement = 0,
				.descriptorCount = 1,
				.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
				.pBufferInfo = &buffer_infos[3],
			},
		};

		vkUpdateDescriptorSets(device, sizeof(write_infos) / sizeof(write_infos[0]),
		                       write_infos, 0, nullptr);
	}

	// NOTE: Plane mesh initialization
	{
		// (v0)------(v1)
		//  |  \       |
		//  |   `--,   |
		//  |       \  |
		// (v3)------(v2)
		std::vector<Vertex> vertices = {
			{{-5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{5.0f, 0.0f, 5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-5.0f, 0.0f, -5.0f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0
		};

		plane_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		plane_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		plane_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Cube mesh initialization
	{
		std::vector<Vertex> vertices = {
			{{-0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, -0.5f}, {0.0f, 0.0f, -1.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{+0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 0.0f}},
			{{-0.5f, -0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
			{{-0.5f, +0.5f, -0.5f}, {-1.0f, 0.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {-1.0f, 0.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, -0.5f, +0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, -0.5f, -0.5f}, {0.0f, -1.0f, 0.0f}, {0.0f, 1.0f}},

			{{-0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
			{{+0.5f, +0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 0.0f}},
			{{+0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {1.0f, 1.0f}},
			{{-0.5f, +0.5f, +0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 1.0f}},
		};

		std::vector<uint32_t> indices = {
			0, 1, 2, 2, 3, 0,
			4, 5, 6, 6, 7, 4,
			8, 9, 10, 10, 11, 8,
			12, 13, 14, 14, 15, 12,
			16, 17, 18, 18, 19, 16,
			20, 21, 22, 22, 23, 20,
		};

		cube_mesh.vertex_buffer = new veekay::graphics::Buffer(
			vertices.size() * sizeof(Vertex), vertices.data(),
			VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);

		cube_mesh.index_buffer = new veekay::graphics::Buffer(
			indices.size() * sizeof(uint32_t), indices.data(),
			VK_BUFFER_USAGE_INDEX_BUFFER_BIT);

		cube_mesh.indices = uint32_t(indices.size());
	}

	// NOTE: Add models to scene
	models.clear();
	auto add_model = [&](const Mesh& mesh, const Transform& transform, const veekay::vec3& albedo,
	                     const veekay::vec3& specular, float shininess) {
		models.emplace_back(Model{
			.mesh = mesh,
			.transform = transform,
			.albedo_color = albedo,
			.specular_color = specular,
			.shininess = shininess,
		});
	};

	const veekay::vec3 matte_specular{0.14f, 0.14f, 0.14f};

	add_model(cube_mesh,
	          Transform{
		          .position = {-1.2f, -0.4f, -0.6f},
	          },
	          veekay::vec3{0.85f, 0.2f, 0.25f},
	          matte_specular,
	          24.0f);

	add_model(cube_mesh,
	          Transform{
		          .position = {1.2f, -0.4f, -0.2f},
	          },
	          veekay::vec3{0.25f, 0.8f, 0.35f},
	          matte_specular,
	          24.0f);

	add_model(cube_mesh,
	          Transform{
		          .position = {0.0f, -0.4f, 1.0f},
	          },
	          veekay::vec3{0.25f, 0.45f, 0.9f},
	          matte_specular,
	          24.0f);

	ambient_light.color = {0.08f, 0.08f, 0.1f};
	ambient_light.intensity = 0.12f;

	directional_light.direction = {0.25f, 1.0f, 0.2f};
	directional_light.color = {1.0f, 0.98f, 0.95f};
	directional_light.intensity = 0.6f;

	point_light_count = 2;
	point_lights_settings[0] = {
		.position = {-2.0f, -2.0f, -2.0f},
		.intensity = 10.0f,
		.color = {1.0f, 0.85f, 0.7f},
	};
	point_lights_settings[1] = {
		.position = {2.0f, -1.6f, 1.6f},
		.intensity = 8.0f,
		.color = {0.7f, 0.85f, 1.0f},
	};

	spot_light_count = 1;
	spot_lights_settings[0] = {
		.position = {0.0f, -1.4f, -3.0f},
		.intensity = 12.0f,
		.direction = {0.0f, 0.6f, 1.0f},
		.inner_angle_degrees = 12.0f,
		.color = {1.0f, 0.95f, 0.9f},
		.outer_angle_degrees = 20.0f,
	};
}

// NOTE: Destroy resources here, do not cause leaks in your program!
void shutdown() {
	VkDevice& device = veekay::app.vk_device;

	vkDestroySampler(device, missing_texture_sampler, nullptr);
	delete missing_texture;

	delete cube_mesh.index_buffer;
	delete cube_mesh.vertex_buffer;

	delete plane_mesh.index_buffer;
	delete plane_mesh.vertex_buffer;

	delete spot_lights_buffer;
	delete point_lights_buffer;

	delete model_uniforms_buffer;
	delete scene_uniforms_buffer;

	vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
	vkDestroyDescriptorPool(device, descriptor_pool, nullptr);

	vkDestroyPipeline(device, pipeline, nullptr);
	vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
	vkDestroyShaderModule(device, fragment_shader_module, nullptr);
	vkDestroyShaderModule(device, vertex_shader_module, nullptr);
}

void update(double time) {
	static double last_time = 0.0;
	static bool time_initialized = false;
	if (!time_initialized) {
		last_time = time;
		time_initialized = true;
	}

	double dt = time - last_time;
	last_time = time;
	float delta = static_cast<float>(dt);

	if (!saved_transform_valid) {
		saved_transform_state = {camera.position, camera.rotation};
		saved_transform_valid = true;
	}
	if (!saved_look_at_valid) {
		saved_look_at_state = {camera.look_at_target, camera.look_at_rotation, camera.look_at_distance};
		saved_look_at_valid = true;
	}

	ImGui::Begin("Camera");
	const char* camera_modes[] = {"Transform", "Look-at"};
	int camera_mode_index = camera.mode == CameraMode::transform ? 0 : 1;
	if (ImGui::Combo("Mode", &camera_mode_index, camera_modes, 2)) {
		CameraMode next_mode = camera_mode_index == 0 ? CameraMode::transform : CameraMode::look_at;
		if (next_mode != camera.mode) {
			if (camera.mode == CameraMode::transform) {
				saved_transform_state = {camera.position, camera.rotation};
				saved_transform_valid = true;
			} else {
				saved_look_at_state = {camera.look_at_target, camera.look_at_rotation, camera.look_at_distance};
				saved_look_at_valid = true;
			}

			camera.mode = next_mode;

			if (camera.mode == CameraMode::transform) {
				if (saved_transform_valid) {
					camera.position = saved_transform_state.position;
					camera.rotation = saved_transform_state.rotation;
				} else {
					CameraTransformState derived = transformFromLookAtState(saved_look_at_state);
					camera.position = derived.position;
					camera.rotation = derived.rotation;
				}
			} else {
				if (saved_look_at_valid) {
					camera.look_at_target = saved_look_at_state.target;
					camera.look_at_rotation = saved_look_at_state.rotation;
					camera.look_at_distance = saved_look_at_state.distance;
				} else {
					CameraLookAtState derived = lookAtFromTransformState(
						saved_transform_state,
						camera.look_at_distance);
					camera.look_at_target = derived.target;
					camera.look_at_rotation = derived.rotation;
					camera.look_at_distance = derived.distance;
				}
			}
		}
	}
	ImGui::SliderFloat("FOV", &camera.fov, 30.0f, 120.0f);
	if (camera.mode == CameraMode::look_at) {
		ImGui::DragFloat3("Target", &camera.look_at_target.x, 0.05f);
		ImGui::DragFloat("Distance", &camera.look_at_distance, 0.05f, 0.5f, 50.0f);
	}
	ImGui::End();

	ImGui::Begin("Lighting");
	ImGui::ColorEdit3("Ambient color", &ambient_light.color.x);
	ImGui::SliderFloat("Ambient intensity", &ambient_light.intensity, 0.0f, 2.0f);
	ImGui::Separator();
	ImGui::ColorEdit3("Directional color", &directional_light.color.x);
	ImGui::SliderFloat("Directional intensity", &directional_light.intensity, 0.0f, 5.0f);
	ImGui::DragFloat3("Directional direction", &directional_light.direction.x, 0.02f);
	ImGui::Separator();

	int point_count = static_cast<int>(point_light_count);
	if (ImGui::SliderInt("Point lights", &point_count, 0, max_point_lights)) {
		point_light_count = static_cast<uint32_t>(point_count);
	}
	for (uint32_t i = 0; i < point_light_count; ++i) {
		char label[32];
		std::snprintf(label, sizeof(label), "Point light %u", i + 1);
		if (ImGui::TreeNode(label)) {
			PointLightSettings& light = point_lights_settings[i];
			ImGui::DragFloat3("Position", &light.position.x, 0.05f);
			ImGui::ColorEdit3("Color", &light.color.x);
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 50.0f);
			ImGui::TreePop();
		}
	}

	int spot_count = static_cast<int>(spot_light_count);
	if (ImGui::SliderInt("Spot lights", &spot_count, 0, max_spot_lights)) {
		spot_light_count = static_cast<uint32_t>(spot_count);
	}
	for (uint32_t i = 0; i < spot_light_count; ++i) {
		char label[32];
		std::snprintf(label, sizeof(label), "Spot light %u", i + 1);
		if (ImGui::TreeNode(label)) {
			SpotLightSettings& light = spot_lights_settings[i];
			ImGui::DragFloat3("Position", &light.position.x, 0.05f);
			ImGui::DragFloat3("Direction", &light.direction.x, 0.02f);
			ImGui::ColorEdit3("Color", &light.color.x);
			ImGui::SliderFloat("Intensity", &light.intensity, 0.0f, 75.0f);
			ImGui::SliderFloat("Inner angle", &light.inner_angle_degrees, 1.0f, 60.0f);
			ImGui::SliderFloat("Outer angle", &light.outer_angle_degrees, 1.0f, 75.0f);
			ImGui::TreePop();
		}
	}
	ImGui::End();

	ImGuiIO& io = ImGui::GetIO();
	if (!io.WantCaptureMouse) {
		using namespace veekay::input;

		if (mouse::isButtonDown(mouse::Button::left)) {
			auto move_delta = mouse::cursorDelta();
			const float sensitivity = 0.0025f;
			float yaw_delta = move_delta.x * sensitivity;
			float pitch_delta = -move_delta.y * sensitivity;

			const float pitch_limit = toRadians(89.0f);
			if (camera.mode == CameraMode::transform) {
				camera.rotation.y += yaw_delta;
				camera.rotation.x = std::clamp(camera.rotation.x + pitch_delta,
				                               -pitch_limit, pitch_limit);
			} else {
				camera.look_at_rotation.y += yaw_delta;
				camera.look_at_rotation.x = std::clamp(camera.look_at_rotation.x + pitch_delta,
				                                       -pitch_limit, pitch_limit);
			}
		}
	}

	if (!io.WantCaptureKeyboard) {
		using namespace veekay::input;

		const float move_speed = 2.5f;
		veekay::vec2 angles = camera.mode == CameraMode::transform
			? veekay::vec2{camera.rotation.x, camera.rotation.y}
			: camera.look_at_rotation;
		veekay::vec3 front = directionFromAngles(angles);
		veekay::vec3 right = veekay::vec3::normalized(
			veekay::vec3::cross(front, veekay::vec3{0.0f, -1.0f, 0.0f}));
		veekay::vec3 up = veekay::vec3::cross(right, front);

		if (camera.mode == CameraMode::transform) {
			if (keyboard::isKeyDown(keyboard::Key::w))
				camera.position += front * move_speed * delta;
			if (keyboard::isKeyDown(keyboard::Key::s))
				camera.position -= front * move_speed * delta;
			if (keyboard::isKeyDown(keyboard::Key::d))
				camera.position += right * move_speed * delta;
			if (keyboard::isKeyDown(keyboard::Key::a))
				camera.position -= right * move_speed * delta;
			if (keyboard::isKeyDown(keyboard::Key::q))
				camera.position += up * move_speed * delta;
			if (keyboard::isKeyDown(keyboard::Key::z))
				camera.position -= up * move_speed * delta;
		} else {
			if (keyboard::isKeyDown(keyboard::Key::w))
				camera.look_at_distance = std::max(0.5f, camera.look_at_distance - move_speed * delta);
			if (keyboard::isKeyDown(keyboard::Key::s))
				camera.look_at_distance += move_speed * delta;
			if (keyboard::isKeyDown(keyboard::Key::d))
				camera.look_at_target += right * move_speed * delta;
			if (keyboard::isKeyDown(keyboard::Key::a))
				camera.look_at_target -= right * move_speed * delta;
			if (keyboard::isKeyDown(keyboard::Key::q))
				camera.look_at_target += up * move_speed * delta;
			if (keyboard::isKeyDown(keyboard::Key::z))
				camera.look_at_target -= up * move_speed * delta;
		}
	}

	float aspect_ratio = float(veekay::app.window_width) / float(veekay::app.window_height);
	veekay::vec3 camera_position = camera.world_position();
	veekay::vec3 directional_direction = directional_light.direction;
	if (veekay::vec3::squaredLength(directional_direction) < 0.0001f) {
		directional_direction = {0.0f, -1.0f, 0.0f};
	}
	directional_direction = veekay::vec3::normalized(directional_direction);

	SceneUniforms scene_uniforms{
		.view_projection = camera.view_projection(aspect_ratio),
		.camera_position = {camera_position.x, camera_position.y, camera_position.z, 1.0f},
		.ambient_color_intensity = {
			ambient_light.color.x,
			ambient_light.color.y,
			ambient_light.color.z,
			ambient_light.intensity,
		},
		.directional_direction_intensity = {
			directional_direction.x,
			directional_direction.y,
			directional_direction.z,
			directional_light.intensity,
		},
		.directional_color = {
			directional_light.color.x,
			directional_light.color.y,
			directional_light.color.z,
			1.0f,
		},
		.light_counts = {
			static_cast<float>(point_light_count),
			static_cast<float>(spot_light_count),
			0.0f,
			0.0f,
		},
	};

	std::vector<ModelUniforms> model_uniforms(models.size());
	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		ModelUniforms& uniforms = model_uniforms[i];

		uniforms.model = model.transform.matrix();
		uniforms.albedo_color = {model.albedo_color.x, model.albedo_color.y, model.albedo_color.z, 1.0f};
		uniforms.specular_color_shininess = {
			model.specular_color.x,
			model.specular_color.y,
			model.specular_color.z,
			model.shininess,
		};
	}

	*(SceneUniforms*)scene_uniforms_buffer->mapped_region = scene_uniforms;

	const size_t alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = model_uniforms.size(); i < n; ++i) {
		const ModelUniforms& uniforms = model_uniforms[i];

		char* const pointer = static_cast<char*>(model_uniforms_buffer->mapped_region) + i * alignment;
		*reinterpret_cast<ModelUniforms*>(pointer) = uniforms;
	}

	std::array<PointLight, max_point_lights> point_lights{};
	for (uint32_t i = 0; i < point_light_count; ++i) {
		const PointLightSettings& settings = point_lights_settings[i];
		point_lights[i] = {
			.position_intensity = {
				settings.position.x,
				settings.position.y,
				settings.position.z,
				settings.intensity,
			},
			.color = {
				settings.color.x,
				settings.color.y,
				settings.color.z,
				1.0f,
			},
		};
	}
	std::memcpy(point_lights_buffer->mapped_region, point_lights.data(), sizeof(point_lights));

	std::array<SpotLight, max_spot_lights> spot_lights{};
	for (uint32_t i = 0; i < spot_light_count; ++i) {
		const SpotLightSettings& settings = spot_lights_settings[i];
		float inner_angle = std::clamp(settings.inner_angle_degrees, 1.0f, 89.0f);
		float outer_angle = std::clamp(settings.outer_angle_degrees, inner_angle, 89.0f);
		veekay::vec3 direction = settings.direction;
		if (veekay::vec3::squaredLength(direction) < 0.0001f) {
			direction = {0.0f, 0.0f, 1.0f};
		}
		direction = veekay::vec3::normalized(direction);

		spot_lights[i] = {
			.position_intensity = {
				settings.position.x,
				settings.position.y,
				settings.position.z,
				settings.intensity,
			},
			.direction_inner = {
				direction.x,
				direction.y,
				direction.z,
				std::cos(toRadians(inner_angle)),
			},
			.color_outer = {
				settings.color.x,
				settings.color.y,
				settings.color.z,
				std::cos(toRadians(outer_angle)),
			},
		};
	}
	std::memcpy(spot_lights_buffer->mapped_region, spot_lights.data(), sizeof(spot_lights));
}

void render(VkCommandBuffer cmd, VkFramebuffer framebuffer) {
	vkResetCommandBuffer(cmd, 0);

	{ // NOTE: Start recording rendering commands
		VkCommandBufferBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
			.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
		};

		vkBeginCommandBuffer(cmd, &info);
	}

	{ // NOTE: Use current swapchain framebuffer and clear it
		VkClearValue clear_color{.color = {{0.1f, 0.1f, 0.1f, 1.0f}}};
		VkClearValue clear_depth{.depthStencil = {1.0f, 0}};

		VkClearValue clear_values[] = {clear_color, clear_depth};

		VkRenderPassBeginInfo info{
			.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO,
			.renderPass = veekay::app.vk_render_pass,
			.framebuffer = framebuffer,
			.renderArea = {
				.extent = {
					veekay::app.window_width,
					veekay::app.window_height
				},
			},
			.clearValueCount = 2,
			.pClearValues = clear_values,
		};

		vkCmdBeginRenderPass(cmd, &info, VK_SUBPASS_CONTENTS_INLINE);
	}

	vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
	VkDeviceSize zero_offset = 0;

	VkBuffer current_vertex_buffer = VK_NULL_HANDLE;
	VkBuffer current_index_buffer = VK_NULL_HANDLE;

	const size_t model_uniorms_alignment =
		veekay::graphics::Buffer::structureAlignment(sizeof(ModelUniforms));

	for (size_t i = 0, n = models.size(); i < n; ++i) {
		const Model& model = models[i];
		const Mesh& mesh = model.mesh;

		if (current_vertex_buffer != mesh.vertex_buffer->buffer) {
			current_vertex_buffer = mesh.vertex_buffer->buffer;
			vkCmdBindVertexBuffers(cmd, 0, 1, &current_vertex_buffer, &zero_offset);
		}

		if (current_index_buffer != mesh.index_buffer->buffer) {
			current_index_buffer = mesh.index_buffer->buffer;
			vkCmdBindIndexBuffer(cmd, current_index_buffer, zero_offset, VK_INDEX_TYPE_UINT32);
		}

		uint32_t offset = i * model_uniorms_alignment;
		vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
		                    0, 1, &descriptor_set, 1, &offset);

		vkCmdDrawIndexed(cmd, mesh.indices, 1, 0, 0, 0);
	}

	vkCmdEndRenderPass(cmd);
	vkEndCommandBuffer(cmd);
}

} // namespace

int main() {
	return veekay::run({
		.init = initialize,
		.shutdown = shutdown,
		.update = update,
		.render = render,
	});
}
