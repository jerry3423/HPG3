#include <tuple>
#include <chrono>
#include <limits>
#include <vector>
#include <stdexcept>

#include <cstdio>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <volk/volk.h>

#if !defined(GLM_FORCE_RADIANS)
#	define GLM_FORCE_RADIANS
#endif
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "../labutils/to_string.hpp"
#include "../labutils/vulkan_window.hpp"

#include "../labutils/angle.hpp"
using namespace labutils::literals;

#include "../labutils/error.hpp"
#include "../labutils/vkutil.hpp"
#include "../labutils/vkimage.hpp"
#include "../labutils/vkobject.hpp"
#include "../labutils/vkbuffer.hpp"
#include "../labutils/allocator.hpp" 
#include "../labutils/to_string.hpp"
namespace lut = labutils;

#include "baked_model.hpp"

namespace
{
	using Clock_ = std::chrono::steady_clock;
	using Secondsf_ = std::chrono::duration<float, std::ratio<1>>;

	namespace cfg
	{
		// Compiled shader code for the graphics pipeline(s)
		// See sources in cw1/shaders/*. 
#		define SHADERDIR_ "assets/cw3/shaders/"
		constexpr char const* kVertShaderPath = SHADERDIR_ "default.vert.spv";
		constexpr char const* kFragShaderPath = SHADERDIR_ "default.frag.spv";
		constexpr char const* kFullVertShaderPath = SHADERDIR_ "fullscreen.vert.spv";
		constexpr char const* kFullFragShaderPath = SHADERDIR_ "fullscreen.frag.spv";
		constexpr char const* kFilterVertShaderPath = SHADERDIR_ "filterbright.vert.spv";
		constexpr char const* kFilterFragShaderPath = SHADERDIR_ "filterbright.frag.spv";
		constexpr char const* kVerticalBlurVertShaderPath = SHADERDIR_ "verticalblur.vert.spv";
		constexpr char const* kVerticalBlurFragShaderPath = SHADERDIR_ "verticalblur.frag.spv";
		constexpr char const* kHorizontalBlurVertShaderPath = SHADERDIR_ "horizontalblur.vert.spv";
		constexpr char const* kHorizontalBlurFragShaderPath = SHADERDIR_ "horizontalblur.frag.spv";
#		undef SHADERDIR_

#		define ASSETDIR_ "assets/cw3/"
		constexpr char const* kPBRPath = ASSETDIR_ "ship.comp5822mesh";
#		undef ASSETDIR_


		// General rule: with a standard 24 bit or 32 bit float depth buffer,
		// you can support a 1:1000 ratio between the near and far plane with
		// minimal depth fighting. Larger ratios will introduce more depth
		// fighting problems; smaller ratios will increase the depth buffer's
		// resolution but will also limit the view distance.
		constexpr float kCameraNear = 0.1f;
		constexpr float kCameraFar = 100.f;

		constexpr auto kCameraFov = 60.0_degf;

		constexpr VkFormat kDepthFormat = VK_FORMAT_D32_SFLOAT;

		constexpr float kCameraBaseSpeed = 1.7f; // units/second
		constexpr float kCameraFastMult = 5.f; // speed multiplier
		constexpr float kCameraSlowMult = 0.05f; // speed multiplier 

		constexpr float kCameraMouseSensitivity = 0.01f; // radians per pixel
	}


	// Local types/structures:
	namespace glsl
	{

		struct Light
		{
			glm::vec4 lightPosition;
			glm::vec4 lightColor;
		};

		struct SceneUniform
		{
			glm::mat4 camera;
			glm::mat4 projection;
			glm::mat4 projCam;
			glm::vec4 cameraPosition;
			Light pointLight[2];

		};

		struct MaterialUniform
		{
			glm::vec4 baseColor;
			glm::vec4 emissiveColor;
			float roughness;
			float metalness;
		};

		static_assert(sizeof(SceneUniform) <= 65536, "SceneUniform must be less than 65536 bytes for vkCmdUpdateBuffer");
		static_assert(sizeof(SceneUniform) % 4 == 0, "SceneUniform size must be a multiple of 4 bytes");
	}

	enum class EInputState
	{
		forward,
		backward,
		strafeLeft,
		strafeRight,
		levitate,
		sink,
		fast,
		slow,
		mousing,
		max
	};

	struct UserState
	{
		bool inputMap[std::size_t(EInputState::max)] = {};

		float mouseX = 0.f, mouseY = 0.f;
		float previousX = 0.f, previousY = 0.f;

		bool wasMousing = false;

		glm::mat4 camera2world = glm::identity<glm::mat4>();
	};

	struct BakedMeshBuffer
	{
		std::vector<lut::Buffer> position;
		std::vector<lut::Buffer> texcoord;
		std::vector<lut::Buffer> normal;
		std::vector<lut::Buffer> indice;
	};

	bool lightFLag = true;

	// Local functions:
	void glfw_callback_key_press(GLFWwindow*, int, int, int, int);

	void glfw_callback_button(GLFWwindow*, int, int, int);

	void glfw_callback_motion(GLFWwindow*, double, double);

	void create_swapchain_framebuffers(lut::VulkanWindow const&, VkRenderPass, std::vector<lut::Framebuffer>&, VkImageView aDepthView);

	void create_intermediate_framebuffers(lut::VulkanWindow const&, VkRenderPass, lut::Framebuffer&, VkImageView aDepthView, VkImageView aColorAttachmentImageView);

	lut::RenderPass create_render_pass_A(lut::VulkanWindow const&);

	lut::RenderPass create_render_pass_B(lut::VulkanWindow const&);

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout, VkDescriptorSetLayout, VkDescriptorSetLayout);

	lut::PipelineLayout create_post_pipeline_layout(lut::VulkanContext const&, VkDescriptorSetLayout, VkDescriptorSetLayout, VkDescriptorSetLayout);

	lut::Pipeline create_baked_model_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	lut::Pipeline create_post_processing_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	lut::Pipeline create_filter_bright_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	lut::Pipeline create_vertical_blur_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	lut::Pipeline create_horizontal_blur_pipeline(lut::VulkanWindow const&, VkRenderPass, VkPipelineLayout);

	void record_commandsA(VkCommandBuffer, VkRenderPass, VkRenderPass, VkFramebuffer, VkPipeline, VkPipeline, VkExtent2D const&,
		BakedModel& ret, VkBuffer aSceneUBO, glsl::SceneUniform const&, VkPipelineLayout, VkDescriptorSet aSceneDescriptors, std::vector<VkDescriptorSet> modelDescriptors,
		BakedMeshBuffer& meshbuffer, std::vector<lut::Buffer> const& aMaterialUBO, std::vector< glsl::MaterialUniform> const& aMaterialUniform, VkDescriptorSet aImageDescriptor,
		VkPipelineLayout, VkPipeline, VkPipeline, VkPipeline, VkFramebuffer, VkDescriptorSet,VkImage filterimage, VkFramebuffer, VkFramebuffer, VkDescriptorSet verticalDescriptor);

	void record_commandsB(VkCommandBuffer, VkRenderPass, VkRenderPass, VkFramebuffer, VkFramebuffer, VkPipeline, VkPipeline, VkExtent2D const&,
		BakedModel& ret, VkPipelineLayout, BakedMeshBuffer& meshbuffer, VkDescriptorSet aImageDescriptor, 
		VkPipelineLayout, VkPipeline, VkPipeline, VkPipeline, VkFramebuffer, VkDescriptorSet, VkDescriptorSet horizontalDescriptor);

	void submit_commands(lut::VulkanContext const&, VkCommandBuffer, VkFence, VkSemaphore, VkSemaphore);

	void present_results(VkQueue, VkSwapchainKHR, std::uint32_t aImageIndex, VkSemaphore, bool& aNeedToRecreateSwapchain);

	void update_scene_uniforms(glsl::SceneUniform&, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState const&, glm::vec4 cameraPos);

	void update_material_uniforms(glsl::MaterialUniform&, glm::vec4 baseColor, glm::vec4 emissiveColor, float roughness, float metalness);

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const&);

	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const&);

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const&, lut::Allocator const&);

	std::tuple<lut::Image, lut::ImageView> create_color_imageview(lut::VulkanWindow const&, lut::Allocator const&);

	glm::vec4 update_user_state(UserState&, float aElapsedTime);

	BakedMeshBuffer upload_mesh_buffer(lut::VulkanContext const&, lut::Allocator const&, BakedModel const& model);

	void updateDescriptorSet(lut::VulkanWindow const&, VkDescriptorSet const&, VkImageView const&, VkSampler const&);
}

int main() try
{
	//create vulkan window
	lut::VulkanWindow window = lut::make_vulkan_window();

	// Configure the GLFW window
	UserState state{};
	glfwSetWindowUserPointer(window.window, &state);

	// Configure the GLFW window
	glfwSetKeyCallback(window.window, &glfw_callback_key_press);
	glfwSetMouseButtonCallback(window.window, &glfw_callback_button);
	glfwSetCursorPosCallback(window.window, &glfw_callback_motion);

	// Create VMA allocator
	lut::Allocator allocator = lut::create_allocator(window);

	//create render pass
	lut::RenderPass renderpassA = create_render_pass_A(window);
	lut::RenderPass renderpassB = create_render_pass_B(window);

	// create scene descriptor set layout
	lut::DescriptorSetLayout sceneLayout = create_scene_descriptor_layout(window);

	// create material descriptor set layout
	lut::DescriptorSetLayout materialLayout = create_scene_descriptor_layout(window);

	//create object descriptor set layout
	lut::DescriptorSetLayout objectLayout = create_object_descriptor_layout(window);

	//create pipeline layout
	lut::PipelineLayout pipeLayout = create_pipeline_layout(window, sceneLayout.handle, objectLayout.handle, materialLayout.handle);
	lut::PipelineLayout postPipeLayout = create_post_pipeline_layout(window, sceneLayout.handle, objectLayout.handle, materialLayout.handle);

	//create pipeline
	lut::Pipeline pipeA = create_baked_model_pipeline(window, renderpassA.handle, pipeLayout.handle);
	lut::Pipeline pipeB = create_post_processing_pipeline(window, renderpassB.handle, postPipeLayout.handle);
	lut::Pipeline filterBrightPipe = create_filter_bright_pipeline(window, renderpassA.handle, pipeLayout.handle);
	lut::Pipeline verticalBlurPipe = create_vertical_blur_pipeline(window, renderpassA.handle, postPipeLayout.handle);
	lut::Pipeline horizontalBlurPipe = create_horizontal_blur_pipeline(window, renderpassA.handle, postPipeLayout.handle);

	//create intermediate image and image view
	auto [colorImage, colorImageView] = create_color_imageview(window, allocator);
	auto [filterImage, filterImageView] = create_color_imageview(window, allocator);
	auto [verticalImage, verticalImageView] = create_color_imageview(window, allocator);
	auto [horizontalImage, horizontalImageView] = create_color_imageview(window, allocator);

	//create depth buffer
	auto [depthBuffer, depthBufferView] = create_depth_buffer(window, allocator);

	//create framebuffer for intermediate view
	lut::Framebuffer intermediateFB;
	create_intermediate_framebuffers(window, renderpassA.handle, intermediateFB, depthBufferView.handle, colorImageView.handle);
	lut::Framebuffer filterframebuffers;
	create_intermediate_framebuffers(window, renderpassA.handle, filterframebuffers, depthBufferView.handle, filterImageView.handle);
	lut::Framebuffer verticalframebuffers;
	create_intermediate_framebuffers(window, renderpassA.handle, verticalframebuffers, depthBufferView.handle, verticalImageView.handle);
	lut::Framebuffer horizontalframebuffers;
	create_intermediate_framebuffers(window, renderpassA.handle, horizontalframebuffers, depthBufferView.handle, horizontalImageView.handle);

	//create framebuffer for swapchain view
	std::vector<lut::Framebuffer> framebuffers;
	create_swapchain_framebuffers(window, renderpassB.handle, framebuffers, depthBufferView.handle);

	//create command pool
	lut::CommandPool cpool = lut::create_command_pool(window, VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT);

	//create command buffer and fence
	std::vector<VkCommandBuffer> cbuffers;
	std::vector<lut::Fence> cbfences;
	for (std::size_t i = 0; i < framebuffers.size(); ++i)
	{
		cbuffers.emplace_back(lut::alloc_command_buffer(window, cpool.handle));
		cbfences.emplace_back(lut::create_fence(window, VK_FENCE_CREATE_SIGNALED_BIT));
	}

	lut::Semaphore imageAvailable = lut::create_semaphore(window);
	lut::Semaphore renderFinished = lut::create_semaphore(window);
	lut::Semaphore intermediateSem = lut::create_semaphore(window);


	//load baked model
	BakedModel model = load_baked_model(cfg::kPBRPath);

	//create scene uniform buffer
	lut::Buffer sceneUBO = lut::create_buffer(allocator, sizeof(glsl::SceneUniform),
		VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
		VMA_MEMORY_USAGE_GPU_ONLY);

	//create descriptor pool
	lut::DescriptorPool dpool = lut::create_descriptor_pool(window);

	//allocate descriptor set for uniform buffer
	//initialize descriptor set with vkUpdateDescriptorSets
	VkDescriptorSet sceneDescriptors = lut::alloc_desc_set(window, dpool.handle, sceneLayout.handle);
	{
		VkWriteDescriptorSet desc[1]{};

		VkDescriptorBufferInfo sceneUboInfo{};
		sceneUboInfo.buffer = sceneUBO.buffer;
		sceneUboInfo.range = VK_WHOLE_SIZE;

		desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		desc[0].dstSet = sceneDescriptors;
		desc[0].dstBinding = 0;
		desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		desc[0].descriptorCount = 1;
		desc[0].pBufferInfo = &sceneUboInfo;

		constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
		vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
	}

	std::vector<lut::Buffer> materialUBO(model.materials.size());

	//create material uniform buffer
	for (size_t i = 0; i < materialUBO.size(); ++i)
	{
		materialUBO[i] = lut::create_buffer(allocator, sizeof(glsl::MaterialUniform),
			VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VMA_MEMORY_USAGE_GPU_ONLY);
	}

	std::vector<VkDescriptorSet> materialDescriptors(model.materials.size());

	for (size_t i = 0; i < materialUBO.size(); ++i)
	{
		materialDescriptors[i] = lut::alloc_desc_set(window, dpool.handle, materialLayout.handle);
		{
			VkWriteDescriptorSet desc[1]{};

			VkDescriptorBufferInfo materialUboInfo{};
			materialUboInfo.buffer = materialUBO[i].buffer;
			materialUboInfo.range = VK_WHOLE_SIZE;

			desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[0].dstSet = materialDescriptors[i];
			desc[0].dstBinding = 0;
			desc[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
			desc[0].descriptorCount = 1;
			desc[0].pBufferInfo = &materialUboInfo;

			constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
			vkUpdateDescriptorSets(window.device, numSets, desc, 0, nullptr);
		}
	}

	//update material uniform
	std::vector<glsl::MaterialUniform> materialUniforms(materialDescriptors.size());
	for (size_t i = 0; i < materialUniforms.size(); ++i)
	{
		update_material_uniforms(materialUniforms[i], glm::vec4(model.materials[i].baseColor, 1.0f), glm::vec4(model.materials[i].emissiveColor, 1.0f),
			model.materials[i].roughness, model.materials[i].metalness);
	}

	//create default texture sampler
	lut::Sampler defaultSampler = lut::create_default_sampler(window);

	VkDescriptorSet ImageDescriptor = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);
	updateDescriptorSet(window, ImageDescriptor, colorImageView.handle, defaultSampler.handle);
	VkDescriptorSet filterDescriptor = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);
	updateDescriptorSet(window, filterDescriptor,filterImageView.handle, defaultSampler.handle);
	VkDescriptorSet verticalDescriptor = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);
	updateDescriptorSet(window, verticalDescriptor, verticalImageView.handle, defaultSampler.handle);
	VkDescriptorSet horizontalDescriptor = lut::alloc_desc_set(window, dpool.handle, objectLayout.handle);
	updateDescriptorSet(window, horizontalDescriptor, horizontalImageView.handle, defaultSampler.handle);

	//Upload mesh to gpu buffer
	BakedMeshBuffer meshbuffer = upload_mesh_buffer(window, allocator, model);

	glm::vec4 cameraPos;

	//add lights to scene
	glsl::SceneUniform sceneUniforms{};
	sceneUniforms.pointLight[0].lightPosition = glm::vec4(0.f, 10.f, 0.f, 1.f);
	sceneUniforms.pointLight[0].lightColor = glm::vec4(1.f, 1.f, 1.f, 1.f);
	sceneUniforms.pointLight[1].lightPosition = glm::vec4(0.f, 10.f, 10.f, 1.f);
	sceneUniforms.pointLight[1].lightColor = glm::vec4(1.f, 1.f, 1.f, 1.f);

	auto previousClock = Clock_::now();

	bool recreateSwapchain = false;
	while (!glfwWindowShouldClose(window.window))
	{
		glfwPollEvents();
		//glfwWaitEvents();

		// Recreate swap chain
		if (recreateSwapchain)
		{
			//re-create swapchain, render pass, framebuffer and pipeline
			vkDeviceWaitIdle(window.device);
			auto const changes = recreate_swapchain(window);

			if (changes.changedFormat)
			{
				renderpassA = create_render_pass_A(window);
				renderpassB = create_render_pass_B(window);
			}
			if (changes.changedSize)
			{
				std::tie(depthBuffer, depthBufferView) = create_depth_buffer(window, allocator);
				std::tie(colorImage, colorImageView) = create_color_imageview(window, allocator);
				std::tie(filterImage, filterImageView) = create_color_imageview(window, allocator);
				std::tie(verticalImage, verticalImageView) = create_color_imageview(window, allocator);
				std::tie(horizontalImage, horizontalImageView) = create_color_imageview(window, allocator);
			}

			create_intermediate_framebuffers(window, renderpassA.handle, intermediateFB, depthBufferView.handle, colorImageView.handle);
			create_intermediate_framebuffers(window, renderpassA.handle, filterframebuffers, depthBufferView.handle, filterImageView.handle);
			create_intermediate_framebuffers(window, renderpassA.handle, verticalframebuffers, depthBufferView.handle, verticalImageView.handle);
			create_intermediate_framebuffers(window, renderpassA.handle, horizontalframebuffers, depthBufferView.handle, horizontalImageView.handle);

			framebuffers.clear();
			create_swapchain_framebuffers(window, renderpassB.handle, framebuffers, depthBufferView.handle);

			updateDescriptorSet(window, ImageDescriptor, colorImageView.handle, defaultSampler.handle);
			updateDescriptorSet(window, filterDescriptor, filterImageView.handle, defaultSampler.handle);
			updateDescriptorSet(window, verticalDescriptor, verticalImageView.handle, defaultSampler.handle);
			updateDescriptorSet(window, horizontalDescriptor, horizontalImageView.handle, defaultSampler.handle);

			if (changes.changedSize)
			{
				pipeA = create_baked_model_pipeline(window, renderpassA.handle, pipeLayout.handle);
				pipeB = create_post_processing_pipeline(window, renderpassB.handle, postPipeLayout.handle);
				filterBrightPipe = create_filter_bright_pipeline(window, renderpassA.handle, pipeLayout.handle);
				verticalBlurPipe = create_vertical_blur_pipeline(window, renderpassA.handle, postPipeLayout.handle);
				horizontalBlurPipe = create_horizontal_blur_pipeline(window, renderpassA.handle, postPipeLayout.handle);
			}

			recreateSwapchain = false;
			continue;
		}

		// Update state
		auto const now = Clock_::now();
		auto const dt = std::chrono::duration_cast<Secondsf_>(now - previousClock).count();
		previousClock = now;
		cameraPos = update_user_state(state, dt);

		// Prepare data for this frame
		update_scene_uniforms(sceneUniforms, window.swapchainExtent.width, window.swapchainExtent.height, state, cameraPos);

		std::uint32_t imageIndex = 0;

		//acquire swapchain image.
		auto const acquireRes = vkAcquireNextImageKHR(
			window.device,
			window.swapchain,
			std::numeric_limits<std::uint64_t>::max(),
			imageAvailable.handle,
			VK_NULL_HANDLE,
			&imageIndex
		);

		//if there is no image in swapchain, recreate it
		if (VK_SUBOPTIMAL_KHR == acquireRes || VK_ERROR_OUT_OF_DATE_KHR == acquireRes)
		{
			recreateSwapchain = true;
			continue;
		}

		//throw error if acquire next image fail
		if (VK_SUCCESS != acquireRes)
		{
			throw lut::Error("Unable to acquire enxt swapchain image\n""vkAcquireNextImageKHR() returned %s", lut::to_string(acquireRes).c_str());
		}

		//wait for command buffer to be available
		assert(std::size_t(imageIndex) < cbfences.size());
		if (auto const res = vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to wait for command buffer fence %u\n""vkWaitForFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}
		if (auto const res = vkResetFences(window.device, 1, &cbfences[imageIndex].handle); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to reset command buffer fence %u\n""vkResetFences() returned %s", imageIndex, lut::to_string(res).c_str());
		}

		//record and submit commands
		assert(std::size_t(imageIndex) < cbuffers.size());
		assert(std::size_t(imageIndex) < framebuffers.size());

		//record and submit command to GPU
		record_commandsA(cbuffers[imageIndex], renderpassA.handle, renderpassB.handle, intermediateFB.handle, pipeA.handle, pipeB.handle, window.swapchainExtent,
			model, sceneUBO.buffer, sceneUniforms, pipeLayout.handle, sceneDescriptors, materialDescriptors, meshbuffer,
			materialUBO, materialUniforms, ImageDescriptor, postPipeLayout.handle, filterBrightPipe.handle, verticalBlurPipe.handle, horizontalBlurPipe.handle,
			filterframebuffers.handle, filterDescriptor, filterImage.image, verticalframebuffers.handle, horizontalframebuffers.handle, verticalDescriptor);

		submit_commands(window, cbuffers[imageIndex], cbfences[imageIndex].handle, imageAvailable.handle, intermediateSem.handle);

		vkWaitForFences(window.device, 1, &cbfences[imageIndex].handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max());
		vkResetFences(window.device, 1, &cbfences[imageIndex].handle);

		//record and submit command to GPU
		record_commandsB(cbuffers[imageIndex], renderpassA.handle, renderpassB.handle, framebuffers[imageIndex].handle, intermediateFB.handle,
			pipeA.handle, pipeB.handle, window.swapchainExtent, model, pipeLayout.handle, meshbuffer,
			ImageDescriptor, postPipeLayout.handle, filterBrightPipe.handle, verticalBlurPipe.handle, horizontalBlurPipe.handle, 
			filterframebuffers.handle, filterDescriptor, horizontalDescriptor);

		submit_commands(window, cbuffers[imageIndex], cbfences[imageIndex].handle, intermediateSem.handle, renderFinished.handle);

		present_results(window.presentQueue, window.swapchain, imageIndex, renderFinished.handle, recreateSwapchain);
	}

	//destroy device after all commands have finished
	vkDeviceWaitIdle(window.device);

	return 0;
}
catch( std::exception const& eErr )
{
	std::fprintf( stderr, "\n" );
	std::fprintf( stderr, "Error: %s\n", eErr.what() );
	return 1;
}

namespace
{
	void glfw_callback_key_press(GLFWwindow* aWindow, int aKey, int /*aScanCode*/, int aAction, int /*aModifierFlags*/)
	{
		if (GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction)
		{
			glfwSetWindowShouldClose(aWindow, GLFW_TRUE);
		}
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWindow));
		assert(state);

		bool const isReleased = (GLFW_RELEASE == aAction);

		switch (aKey)
		{
		case GLFW_KEY_W:
			state->inputMap[std::size_t(EInputState::forward)] = !isReleased;
			break;
		case GLFW_KEY_S:
			state->inputMap[std::size_t(EInputState::backward)] = !isReleased;
			break;
		case GLFW_KEY_A:
			state->inputMap[std::size_t(EInputState::strafeLeft)] = !isReleased;
			break;
		case GLFW_KEY_D:
			state->inputMap[std::size_t(EInputState::strafeRight)] = !isReleased;
			break;
		case GLFW_KEY_E:
			state->inputMap[std::size_t(EInputState::levitate)] = !isReleased;
			break;
		case GLFW_KEY_Q:
			state->inputMap[std::size_t(EInputState::sink)] = !isReleased;
			break;
		case GLFW_KEY_P:
			lightFLag = false;
			break;
		case GLFW_KEY_O:
			lightFLag = true;
			break;
		case GLFW_KEY_LEFT_SHIFT: [[fallthrough]];
		case GLFW_KEY_RIGHT_SHIFT:
			state->inputMap[std::size_t(EInputState::fast)] = !isReleased;
			break;

		case GLFW_KEY_LEFT_CONTROL: [[fallthrough]];
		case GLFW_KEY_RIGHT_CONTROL:
			state->inputMap[std::size_t(EInputState::slow)] = !isReleased;
			break;

		default:
			break;
		}
	}

	void glfw_callback_button(GLFWwindow* aWin, int aBut, int aAct, int)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		if (GLFW_MOUSE_BUTTON_RIGHT == aBut && GLFW_PRESS == aAct)
		{
			auto& flag = state->inputMap[std::size_t(EInputState::mousing)];

			flag = !flag;
			if (flag)
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else
				glfwSetInputMode(aWin, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}

	void glfw_callback_motion(GLFWwindow* aWin, double aX, double aY)
	{
		auto state = static_cast<UserState*>(glfwGetWindowUserPointer(aWin));
		assert(state);

		state->mouseX = float(aX);
		state->mouseY = float(aY);
	}

	void create_swapchain_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, std::vector<lut::Framebuffer>& aFramebuffers,
		VkImageView aDepthView)
	{
		assert(aFramebuffers.empty());

		for (std::size_t i = 0; i < aWindow.swapViews.size(); ++i)
		{
			VkImageView attachments[2]{
				aWindow.swapViews[i],
				aDepthView
			};

			VkFramebufferCreateInfo FramebufferInfo{};
			FramebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			FramebufferInfo.flags = 0;
			FramebufferInfo.renderPass = aRenderPass;
			FramebufferInfo.attachmentCount = 2;
			FramebufferInfo.pAttachments = attachments;
			FramebufferInfo.width = aWindow.swapchainExtent.width;
			FramebufferInfo.height = aWindow.swapchainExtent.height;
			FramebufferInfo.layers = 1;

			VkFramebuffer fb = VK_NULL_HANDLE;
			if (auto const res = vkCreateFramebuffer(aWindow.device, &FramebufferInfo, nullptr, &fb); VK_SUCCESS != res)
			{
				throw lut::Error("Unable to create framebuffer\n""vkCreateFramebuffer() returned %s", lut::to_string(res).c_str());
			}
			aFramebuffers.emplace_back(lut::Framebuffer(aWindow.device, fb));
		}
	}

	void create_intermediate_framebuffers(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, lut::Framebuffer& aFramebuffers,
		VkImageView aDepthView, VkImageView aColorAttachmentImageView)
	{
		VkImageView attachments[2]{
				aColorAttachmentImageView,
				aDepthView
		};

		VkFramebufferCreateInfo FramebufferInfo{};
		FramebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
		FramebufferInfo.flags = 0;
		FramebufferInfo.renderPass = aRenderPass;
		FramebufferInfo.attachmentCount = 2;
		FramebufferInfo.pAttachments = attachments;
		FramebufferInfo.width = aWindow.swapchainExtent.width;
		FramebufferInfo.height = aWindow.swapchainExtent.height;
		FramebufferInfo.layers = 1;

		VkFramebuffer fb = VK_NULL_HANDLE;
		if (auto const res = vkCreateFramebuffer(aWindow.device, &FramebufferInfo, nullptr, &fb); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create framebuffer\n""vkCreateFramebuffer() returned %s", lut::to_string(res).c_str());
		}
		aFramebuffers = lut::Framebuffer(aWindow.device, fb);
	}

	lut::RenderPass create_render_pass_A(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1; // this refers to attachments[1]
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;

		VkSubpassDependency dependencies[3]{};
		dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[0].dstSubpass = 0;
		dependencies[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[0].dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[0].dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[0].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[1].srcSubpass = 0;
		dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[1].srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		dependencies[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		dependencies[1].srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
		dependencies[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
		dependencies[1].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		dependencies[2].srcSubpass = VK_SUBPASS_EXTERNAL;
		dependencies[2].dstSubpass = 0;
		dependencies[2].srcStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[2].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT | VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
		dependencies[2].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[2].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
		dependencies[2].dependencyFlags = VK_DEPENDENCY_BY_REGION_BIT;

		VkRenderPassCreateInfo passinfo{};
		passinfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passinfo.attachmentCount = 2;
		passinfo.pAttachments = attachments;
		passinfo.subpassCount = 1;
		passinfo.pSubpasses = subpasses;
		passinfo.dependencyCount = 3;
		passinfo.pDependencies = dependencies;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passinfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n""vkCreateRenderPass() returned % s", lut::to_string(res).c_str());
		}


		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::RenderPass create_render_pass_B(lut::VulkanWindow const& aWindow)
	{
		VkAttachmentDescription attachments[2]{};
		attachments[0].format = aWindow.swapchainFormat;
		attachments[0].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[0].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[0].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		attachments[0].initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		attachments[0].finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		attachments[1].format = cfg::kDepthFormat;
		attachments[1].samples = VK_SAMPLE_COUNT_1_BIT;
		attachments[1].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		attachments[1].storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		attachments[1].initialLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
		attachments[1].finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference subpassAttachments[1]{};
		subpassAttachments[0].attachment = 0;
		subpassAttachments[0].layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depthAttachment{};
		depthAttachment.attachment = 1; // this refers to attachments[1]
		depthAttachment.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpasses[1]{};
		subpasses[0].pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpasses[0].colorAttachmentCount = 1;
		subpasses[0].pColorAttachments = subpassAttachments;
		subpasses[0].pDepthStencilAttachment = &depthAttachment;


		VkRenderPassCreateInfo passinfo{};
		passinfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		passinfo.attachmentCount = 2;
		passinfo.pAttachments = attachments;
		passinfo.subpassCount = 1;
		passinfo.pSubpasses = subpasses;
		passinfo.dependencyCount = 0;
		passinfo.pDependencies = nullptr;

		VkRenderPass rpass = VK_NULL_HANDLE;
		if (auto const res = vkCreateRenderPass(aWindow.device, &passinfo, nullptr, &rpass); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create render pass\n""vkCreateRenderPass() returned % s", lut::to_string(res).c_str());
		}

		return lut::RenderPass(aWindow.device, rpass);
	}

	lut::PipelineLayout create_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout,
		VkDescriptorSetLayout materialLayout)
	{
		VkDescriptorSetLayout layouts[] = {
			aSceneLayout,
			materialLayout
		};
		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::PipelineLayout create_post_pipeline_layout(lut::VulkanContext const& aContext, VkDescriptorSetLayout aSceneLayout, VkDescriptorSetLayout aObjectLayout,
		VkDescriptorSetLayout materialLayout)
	{
		VkDescriptorSetLayout layouts[] = {
			aObjectLayout,
			aObjectLayout
		};
		VkPipelineLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		layoutInfo.setLayoutCount = sizeof(layouts) / sizeof(layouts[0]);
		layoutInfo.pSetLayouts = layouts;
		layoutInfo.pushConstantRangeCount = 0;
		layoutInfo.pPushConstantRanges = nullptr;

		VkPipelineLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreatePipelineLayout(aContext.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create pipeline layout\n""vkCreatePipelineLayout() returned %s", lut::to_string(res).c_str());
		}

		return lut::PipelineLayout(aContext.device, layout);
	}

	lut::Pipeline create_baked_model_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFragShaderPath);

		// Define shader stages
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkVertexInputBindingDescription vertexInputs[3]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[3]{};
		vertexAttributes[0].binding = 0; // must match binding above
		vertexAttributes[0].location = 0; // must match shader
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above
		vertexAttributes[1].location = 1; // must match shader
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2; // must match binding above
		vertexAttributes[2].location = 2; // must match shader
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 3; // number of vertexInputs above
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 3; // number of vertexAttributes above
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		// Input assembly
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Tessellation state(...)
		// Viewport state
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Rasterization state
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.f;

		// Multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Depth state
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		// Blend state
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Dynamic state
		VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

		VkPipelineDynamicStateCreateInfo dynamicInfo = {};
		dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicInfo.dynamicStateCount = 2;
		dynamicInfo.pDynamicStates = dynamicStates;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pDynamicState = &dynamicInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pRasterizationState = &rasterInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n""vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_post_processing_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kFullVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFullFragShaderPath);

		// Define shader stages
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 0; // number of vertexInputs above
		inputInfo.pVertexBindingDescriptions = nullptr;
		inputInfo.vertexAttributeDescriptionCount = 0; // number of vertexAttributes above
		inputInfo.pVertexAttributeDescriptions = nullptr;

		// Input assembly
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Tessellation state(...)
		// Viewport state
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Rasterization state
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.f;

		// Multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Depth state
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_FALSE;
		depthInfo.depthWriteEnable = VK_FALSE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		// Blend state
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Dynamic state
		VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

		VkPipelineDynamicStateCreateInfo dynamicInfo = {};
		dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicInfo.dynamicStateCount = 2;
		dynamicInfo.pDynamicStates = dynamicStates;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pDynamicState = &dynamicInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pRasterizationState = &rasterInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n""vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_filter_bright_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kFilterVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kFilterFragShaderPath);

		// Define shader stages
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";

		VkVertexInputBindingDescription vertexInputs[3]{};
		vertexInputs[0].binding = 0;
		vertexInputs[0].stride = sizeof(float) * 3;
		vertexInputs[0].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[1].binding = 1;
		vertexInputs[1].stride = sizeof(float) * 2;
		vertexInputs[1].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		vertexInputs[2].binding = 2;
		vertexInputs[2].stride = sizeof(float) * 3;
		vertexInputs[2].inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		VkVertexInputAttributeDescription vertexAttributes[3]{};
		vertexAttributes[0].binding = 0; // must match binding above
		vertexAttributes[0].location = 0; // must match shader
		vertexAttributes[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[0].offset = 0;

		vertexAttributes[1].binding = 1; // must match binding above
		vertexAttributes[1].location = 1; // must match shader
		vertexAttributes[1].format = VK_FORMAT_R32G32_SFLOAT;
		vertexAttributes[1].offset = 0;

		vertexAttributes[2].binding = 2; // must match binding above
		vertexAttributes[2].location = 2; // must match shader
		vertexAttributes[2].format = VK_FORMAT_R32G32B32_SFLOAT;
		vertexAttributes[2].offset = 0;


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 3; // number of vertexInputs above
		inputInfo.pVertexBindingDescriptions = vertexInputs;
		inputInfo.vertexAttributeDescriptionCount = 3; // number of vertexAttributes above
		inputInfo.pVertexAttributeDescriptions = vertexAttributes;

		// Input assembly
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Tessellation state(...)
		// Viewport state
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Rasterization state
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.f;

		// Multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Depth state
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		// Blend state
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Dynamic state
		VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

		VkPipelineDynamicStateCreateInfo dynamicInfo = {};
		dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicInfo.dynamicStateCount = 2;
		dynamicInfo.pDynamicStates = dynamicStates;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pDynamicState = &dynamicInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pRasterizationState = &rasterInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n""vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_vertical_blur_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kVerticalBlurVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kVerticalBlurFragShaderPath);

		// Define shader stages
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 0; // number of vertexInputs above
		inputInfo.pVertexBindingDescriptions = nullptr;
		inputInfo.vertexAttributeDescriptionCount = 0; // number of vertexAttributes above
		inputInfo.pVertexAttributeDescriptions = nullptr;

		// Input assembly
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Tessellation state(...)
		// Viewport state
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Rasterization state
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.f;

		// Multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Depth state
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		// Blend state
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Dynamic state
		VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

		VkPipelineDynamicStateCreateInfo dynamicInfo = {};
		dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicInfo.dynamicStateCount = 2;
		dynamicInfo.pDynamicStates = dynamicStates;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pDynamicState = &dynamicInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pRasterizationState = &rasterInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n""vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}

	lut::Pipeline create_horizontal_blur_pipeline(lut::VulkanWindow const& aWindow, VkRenderPass aRenderPass, VkPipelineLayout aPipelineLayout)
	{
		lut::ShaderModule vert = lut::load_shader_module(aWindow, cfg::kHorizontalBlurVertShaderPath);
		lut::ShaderModule frag = lut::load_shader_module(aWindow, cfg::kHorizontalBlurFragShaderPath);

		// Define shader stages
		VkPipelineShaderStageCreateInfo stages[2]{};
		stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
		stages[0].module = vert.handle;
		stages[0].pName = "main";

		stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		stages[1].module = frag.handle;
		stages[1].pName = "main";


		VkPipelineVertexInputStateCreateInfo inputInfo{};
		inputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		inputInfo.vertexBindingDescriptionCount = 0; // number of vertexInputs above
		inputInfo.pVertexBindingDescriptions = nullptr;
		inputInfo.vertexAttributeDescriptionCount = 0; // number of vertexAttributes above
		inputInfo.pVertexAttributeDescriptions = nullptr;

		// Input assembly
		VkPipelineInputAssemblyStateCreateInfo assemblyInfo{};
		assemblyInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		assemblyInfo.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		assemblyInfo.primitiveRestartEnable = VK_FALSE;

		// Tessellation state(...)
		// Viewport state
		VkViewport viewport{};
		viewport.x = 0.f;
		viewport.y = 0.f;
		viewport.width = float(aWindow.swapchainExtent.width);
		viewport.height = float(aWindow.swapchainExtent.height);
		viewport.minDepth = 0.f;
		viewport.maxDepth = 1.f;

		VkRect2D scissor{};
		scissor.offset = VkOffset2D{ 0,0 };
		scissor.extent = VkExtent2D{ aWindow.swapchainExtent.width, aWindow.swapchainExtent.height };

		VkPipelineViewportStateCreateInfo viewportInfo{};
		viewportInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewportInfo.viewportCount = 1;
		viewportInfo.pViewports = &viewport;
		viewportInfo.scissorCount = 1;
		viewportInfo.pScissors = &scissor;

		// Rasterization state
		VkPipelineRasterizationStateCreateInfo rasterInfo{};
		rasterInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterInfo.depthBiasEnable = VK_FALSE;
		rasterInfo.depthClampEnable = VK_FALSE;
		rasterInfo.rasterizerDiscardEnable = VK_FALSE;
		rasterInfo.polygonMode = VK_POLYGON_MODE_FILL;
		rasterInfo.cullMode = VK_CULL_MODE_FRONT_BIT;
		rasterInfo.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
		rasterInfo.lineWidth = 1.f;

		// Multisampling state
		VkPipelineMultisampleStateCreateInfo samplingInfo{};
		samplingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		samplingInfo.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

		// Depth state
		VkPipelineDepthStencilStateCreateInfo depthInfo{};
		depthInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depthInfo.depthTestEnable = VK_TRUE;
		depthInfo.depthWriteEnable = VK_TRUE;
		depthInfo.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;
		depthInfo.minDepthBounds = 0.f;
		depthInfo.maxDepthBounds = 1.f;

		// Blend state
		VkPipelineColorBlendAttachmentState blendStates[1]{};
		blendStates[0].blendEnable = VK_FALSE;
		blendStates[0].colorBlendOp = VK_BLEND_OP_ADD;
		blendStates[0].srcColorBlendFactor = VK_BLEND_FACTOR_SRC_ALPHA;
		blendStates[0].dstColorBlendFactor = VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA;
		blendStates[0].colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

		VkPipelineColorBlendStateCreateInfo blendInfo{};
		blendInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		blendInfo.logicOpEnable = VK_FALSE;
		blendInfo.attachmentCount = 1;
		blendInfo.pAttachments = blendStates;

		// Dynamic state
		VkDynamicState dynamicStates[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };

		VkPipelineDynamicStateCreateInfo dynamicInfo = {};
		dynamicInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamicInfo.dynamicStateCount = 2;
		dynamicInfo.pDynamicStates = dynamicStates;

		// Create pipeline
		VkGraphicsPipelineCreateInfo pipeInfo{};
		pipeInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		pipeInfo.stageCount = 2;
		pipeInfo.pStages = stages;
		pipeInfo.pVertexInputState = &inputInfo;
		pipeInfo.pInputAssemblyState = &assemblyInfo;
		pipeInfo.pTessellationState = nullptr;
		pipeInfo.pViewportState = &viewportInfo;
		pipeInfo.pDepthStencilState = &depthInfo;
		pipeInfo.pDynamicState = &dynamicInfo;
		pipeInfo.pColorBlendState = &blendInfo;
		pipeInfo.pMultisampleState = &samplingInfo;
		pipeInfo.pRasterizationState = &rasterInfo;

		pipeInfo.layout = aPipelineLayout;
		pipeInfo.renderPass = aRenderPass;
		pipeInfo.subpass = 0;

		VkPipeline pipe = VK_NULL_HANDLE;
		if (auto const res = vkCreateGraphicsPipelines(aWindow.device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create graphics pipeline\n""vkCreateGraphicsPipelines() returned %s", lut::to_string(res).c_str());
		}

		return lut::Pipeline(aWindow.device, pipe);
	}


	void record_commandsA(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPassA, VkRenderPass aRenderPassB,  VkFramebuffer interFramebuffer,
		VkPipeline PipeA, VkPipeline PipeB, VkExtent2D const& aImageExtent, BakedModel& model, VkBuffer aSceneUBO, glsl::SceneUniform const& aSceneUniform, 
		VkPipelineLayout aGraphicsLayout, VkDescriptorSet aSceneDescriptors,std::vector<VkDescriptorSet> materialDescriptors, BakedMeshBuffer& meshbuffer, 
		std::vector<lut::Buffer> const& aMaterialUBO, std::vector<glsl::MaterialUniform> const& aMaterialUniform, VkDescriptorSet aImageDescriptor, VkPipelineLayout postPipeLayout, 
		VkPipeline filterPipe, VkPipeline verticalPipe, VkPipeline horizontalPipe, VkFramebuffer filterFB, VkDescriptorSet aFilterDescriptor, VkImage filterimage, VkFramebuffer verticalfb,
		VkFramebuffer horizontalfb, VkDescriptorSet verticalDescriptor)
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &beginInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n""vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		// Upload scene uniforms
		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT
		);

		vkCmdUpdateBuffer(aCmdBuff, aSceneUBO, 0, sizeof(glsl::SceneUniform), &aSceneUniform);

		lut::buffer_barrier(aCmdBuff,
			aSceneUBO,
			VK_ACCESS_TRANSFER_WRITE_BIT,
			VK_ACCESS_UNIFORM_READ_BIT,
			VK_PIPELINE_STAGE_TRANSFER_BIT,
			VK_PIPELINE_STAGE_VERTEX_SHADER_BIT
		);

		for (size_t i = 0; i < materialDescriptors.size(); ++i)
		{
			// Upload material uniforms
			lut::buffer_barrier(aCmdBuff,
				aMaterialUBO[i].buffer,
				VK_ACCESS_UNIFORM_READ_BIT,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT
			);

			vkCmdUpdateBuffer(aCmdBuff, aMaterialUBO[i].buffer, 0, sizeof(glsl::MaterialUniform), &aMaterialUniform[i]);

			lut::buffer_barrier(aCmdBuff,
				aMaterialUBO[i].buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_UNIFORM_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT
			);
		}

		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.0f;
		clearValues[0].color.float32[1] = 0.0f;
		clearValues[0].color.float32[2] = 0.0f;
		clearValues[0].color.float32[3] = 1.f;

		clearValues[1].depthStencil.depth = 1.f;//for depth test

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)aImageExtent.width;
		viewport.height = (float)aImageExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = aImageExtent;

		vkCmdSetViewport(aCmdBuff, 0, 1, &viewport);
		vkCmdSetScissor(aCmdBuff, 0, 1, &scissor);

		//filter bright
		VkRenderPassBeginInfo passInfofilter{};
		passInfofilter.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfofilter.renderPass = aRenderPassA;
		passInfofilter.framebuffer = filterFB;
		passInfofilter.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfofilter.renderArea.extent = VkExtent2D{ aImageExtent.width , aImageExtent.height };
		passInfofilter.clearValueCount = 2;
		passInfofilter.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfofilter, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, filterPipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);
		for (size_t i = 0; i < model.meshes.size(); i++)
		{
			BakedMeshData mesh = model.meshes[i];

			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, &materialDescriptors[mesh.materialId], 0, nullptr);

			// Bind vertex and index buffers
			VkBuffer vertexBuffers[3] = { meshbuffer.position[i].buffer, meshbuffer.texcoord[i].buffer, meshbuffer.normal[i].buffer };
			VkDeviceSize offsets[3] = { 0, 0, 0 };
			vkCmdBindVertexBuffers(aCmdBuff, 0, 3, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(aCmdBuff, meshbuffer.indice[i].buffer, 0, VK_INDEX_TYPE_UINT32);

			// Draw indexed
			vkCmdDrawIndexed(aCmdBuff, static_cast<std::uint32_t>(model.meshes[i].indices.size()), 1, 0, 0, 0);
		}
		vkCmdEndRenderPass(aCmdBuff);

		passInfofilter.framebuffer = verticalfb;
		vkCmdBeginRenderPass(aCmdBuff, &passInfofilter, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, verticalPipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipeLayout, 0, 1, &aFilterDescriptor, 0, nullptr);
		vkCmdDraw(aCmdBuff, 3, 1, 0, 0);
		vkCmdEndRenderPass(aCmdBuff);

		passInfofilter.framebuffer = horizontalfb;
		vkCmdBeginRenderPass(aCmdBuff, &passInfofilter, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, horizontalPipe);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipeLayout, 0, 1, &verticalDescriptor, 0, nullptr);
		vkCmdDraw(aCmdBuff, 3, 1, 0, 0);
		vkCmdEndRenderPass(aCmdBuff);

		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.f;

		//begin renderpass A
		VkRenderPassBeginInfo passInfoA{};
		passInfoA.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfoA.renderPass = aRenderPassA;
		passInfoA.framebuffer = interFramebuffer;
		passInfoA.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfoA.renderArea.extent = VkExtent2D{ aImageExtent.width , aImageExtent.height };
		passInfoA.clearValueCount = 2;
		passInfoA.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfoA, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeA);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 0, 1, &aSceneDescriptors, 0, nullptr);
		for (size_t i = 0; i < model.meshes.size(); i++)
		{
			BakedMeshData mesh = model.meshes[i];

			vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, aGraphicsLayout, 1, 1, &materialDescriptors[mesh.materialId], 0, nullptr);

			// Bind vertex and index buffers
			VkBuffer vertexBuffers[3] = { meshbuffer.position[i].buffer, meshbuffer.texcoord[i].buffer, meshbuffer.normal[i].buffer};
			VkDeviceSize offsets[3] = { 0, 0, 0};
			vkCmdBindVertexBuffers(aCmdBuff, 0, 3, vertexBuffers, offsets);
			vkCmdBindIndexBuffer(aCmdBuff, meshbuffer.indice[i].buffer, 0, VK_INDEX_TYPE_UINT32);

			// Draw indexed
			vkCmdDrawIndexed(aCmdBuff, static_cast<std::uint32_t>(model.meshes[i].indices.size()), 1, 0, 0, 0);
		}
		vkCmdEndRenderPass(aCmdBuff);

		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n""vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}
	}


	void record_commandsB(VkCommandBuffer aCmdBuff, VkRenderPass aRenderPassA, VkRenderPass aRenderPassB, VkFramebuffer aFramebuffer, VkFramebuffer interFramebuffer,
		VkPipeline PipeA, VkPipeline PipeB, VkExtent2D const& aImageExtent, BakedModel& model,
		VkPipelineLayout aGraphicsLayout, BakedMeshBuffer& meshbuffer, VkDescriptorSet aImageDescriptor, VkPipelineLayout postPipeLayout,
		VkPipeline filterPipe, VkPipeline verticalPipe, VkPipeline horizontalPipe, VkFramebuffer filterFB, VkDescriptorSet aFilterDescriptor, VkDescriptorSet horizontalDescriptor)
	{
		VkCommandBufferBeginInfo beginInfo{};
		beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
		beginInfo.pInheritanceInfo = nullptr;

		if (auto const res = vkBeginCommandBuffer(aCmdBuff, &beginInfo); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to begin recording command buffer\n""vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
		}

		VkClearValue clearValues[2]{};
		clearValues[0].color.float32[0] = 0.1f;
		clearValues[0].color.float32[1] = 0.1f;
		clearValues[0].color.float32[2] = 0.1f;
		clearValues[0].color.float32[3] = 1.f;

		clearValues[1].depthStencil.depth = 1.f;//for depth test

		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = (float)aImageExtent.width;
		viewport.height = (float)aImageExtent.height;
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor = {};
		scissor.offset = { 0, 0 };
		scissor.extent = aImageExtent;

		vkCmdSetViewport(aCmdBuff, 0, 1, &viewport);
		vkCmdSetScissor(aCmdBuff, 0, 1, &scissor);

		//begin renderpass B
		VkRenderPassBeginInfo passInfoB{};
		passInfoB.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		passInfoB.renderPass = aRenderPassB;
		passInfoB.framebuffer = aFramebuffer;
		passInfoB.renderArea.offset = VkOffset2D{ 0, 0 };
		passInfoB.renderArea.extent = VkExtent2D{ aImageExtent.width , aImageExtent.height };
		passInfoB.clearValueCount = 2;
		passInfoB.pClearValues = clearValues;

		vkCmdBeginRenderPass(aCmdBuff, &passInfoB, VK_SUBPASS_CONTENTS_INLINE);

		vkCmdBindPipeline(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, PipeB);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipeLayout, 0, 1, &horizontalDescriptor, 0, nullptr);
		vkCmdBindDescriptorSets(aCmdBuff, VK_PIPELINE_BIND_POINT_GRAPHICS, postPipeLayout, 1, 1, &aImageDescriptor, 0, nullptr);
		vkCmdDraw(aCmdBuff, 3, 1, 0, 0);

		vkCmdEndRenderPass(aCmdBuff);

		if (auto const res = vkEndCommandBuffer(aCmdBuff); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to end recording command buffer\n""vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
		}
	}


	void submit_commands(lut::VulkanContext const& aContext, VkCommandBuffer aCmdBuff, VkFence aFence, VkSemaphore aWaitSemaphore, VkSemaphore aSignalSemaphore)
	{
		VkPipelineStageFlags waitPipelineStages = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		VkSubmitInfo submitInfo{};
		submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &aCmdBuff;
		submitInfo.waitSemaphoreCount = 1;
		submitInfo.pWaitSemaphores = &aWaitSemaphore;
		submitInfo.pWaitDstStageMask = &waitPipelineStages;
		submitInfo.signalSemaphoreCount = 1;
		submitInfo.pSignalSemaphores = &aSignalSemaphore;

		if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, aFence); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to submit command buffer to queue\n""vkQueueSubmit() returned %s", lut::to_string(res).c_str());
		}
	}

	void present_results(VkQueue aPresentQueue, VkSwapchainKHR aSwapchain, std::uint32_t aImageIndex, VkSemaphore aRenderFinished, bool& aNeedToRecreateSwapchain)
	{
		VkPresentInfoKHR presentInfo{};
		presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		presentInfo.waitSemaphoreCount = 1;
		presentInfo.pWaitSemaphores = &aRenderFinished;
		presentInfo.swapchainCount = 1;
		presentInfo.pSwapchains = &aSwapchain;
		presentInfo.pImageIndices = &aImageIndex;
		presentInfo.pResults = nullptr;

		auto const presentRes = vkQueuePresentKHR(aPresentQueue, &presentInfo);

		if (VK_SUBOPTIMAL_KHR == presentRes || VK_ERROR_OUT_OF_DATE_KHR == presentRes)
		{
			aNeedToRecreateSwapchain = true;
		}
		else if (VK_SUCCESS != presentRes)
		{
			throw lut::Error("Unable present swapchain image %u\n""vkQueuePresentKHR() returned %s", aImageIndex, lut::to_string(presentRes).c_str());
		}
	}

	lut::DescriptorSetLayout create_scene_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};
		bindings[0].binding = 0;
		bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		bindings[0].descriptorCount = 1;
		bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::DescriptorSetLayout(aWindow.device, layout);
	}

	lut::DescriptorSetLayout create_object_descriptor_layout(lut::VulkanWindow const& aWindow)
	{
		VkDescriptorSetLayoutBinding bindings[1]{};

		for (auto i = 0; i < 1; ++i)
		{
			bindings[i].binding = i;
			bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			bindings[i].descriptorCount = 1;
			bindings[i].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
		}

		VkDescriptorSetLayoutCreateInfo layoutInfo{};
		layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		layoutInfo.bindingCount = sizeof(bindings) / sizeof(bindings[0]);
		layoutInfo.pBindings = bindings;

		VkDescriptorSetLayout layout = VK_NULL_HANDLE;
		if (auto const res = vkCreateDescriptorSetLayout(aWindow.device, &layoutInfo, nullptr, &layout); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create descriptor set layout\n""vkCreateDescriptorSetLayout() returned %s", lut::to_string(res).c_str());
		}
		return lut::DescriptorSetLayout(aWindow.device, layout);
	}


	void update_scene_uniforms(glsl::SceneUniform& aSceneUniforms, std::uint32_t aFramebufferWidth, std::uint32_t aFramebufferHeight, UserState const& aState,
		glm::vec4 cameraPos)
	{
		float const aspect = aFramebufferWidth / float(aFramebufferHeight);

		aSceneUniforms.projection = glm::perspectiveRH_ZO(lut::Radians(cfg::kCameraFov).value(),
			aspect,
			cfg::kCameraNear,
			cfg::kCameraFar
		);
		aSceneUniforms.projection[1][1] *= -1.f; // mirror Y axis

		//aSceneUniforms.camera = glm::translate(glm::vec3(0.f, -0.3f, -1.f));
		aSceneUniforms.camera = glm::inverse(aState.camera2world);
		aSceneUniforms.projCam = aSceneUniforms.projection * aSceneUniforms.camera;
		aSceneUniforms.cameraPosition = cameraPos;
	}

	void update_material_uniforms(glsl::MaterialUniform& aMaterialUniforms, glm::vec4 baseColor, glm::vec4 emissiveColor, float roughness, float metalness)
	{
		aMaterialUniforms.baseColor = baseColor;
		aMaterialUniforms.emissiveColor = emissiveColor;
		aMaterialUniforms.roughness = roughness;
		aMaterialUniforms.metalness = metalness;
	}

	std::tuple<lut::Image, lut::ImageView> create_depth_buffer(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = cfg::kDepthFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n""vmaCreateImage() returned %s", lut::to_string(res).c_str());
		}

		lut::Image depthImage(aAllocator.allocator, image, allocation);
		// Create the image view
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = depthImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = cfg::kDepthFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_DEPTH_BIT,
			0, 1,
			0, 1
		};
		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n""vkCreateImageView() returned %s", lut::to_string(res).c_str());
		}
		return{ std::move(depthImage), lut::ImageView(aWindow.device, view) };
	}

	std::tuple<lut::Image, lut::ImageView> create_color_imageview(lut::VulkanWindow const& aWindow, lut::Allocator const& aAllocator)
	{
		VkImageCreateInfo imageInfo{};
		imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		imageInfo.imageType = VK_IMAGE_TYPE_2D;
		imageInfo.format = aWindow.swapchainFormat;
		imageInfo.extent.width = aWindow.swapchainExtent.width;
		imageInfo.extent.height = aWindow.swapchainExtent.height;
		imageInfo.extent.depth = 1;
		imageInfo.mipLevels = 1;
		imageInfo.arrayLayers = 1;
		imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
		imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
		imageInfo.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
		imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

		VmaAllocationCreateInfo allocInfo{};
		allocInfo.usage = VMA_MEMORY_USAGE_GPU_ONLY;
		VkImage image = VK_NULL_HANDLE;
		VmaAllocation allocation = VK_NULL_HANDLE;

		if (auto const res = vmaCreateImage(aAllocator.allocator, &imageInfo, &allocInfo, &image, &allocation, nullptr); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to allocate depth buffer image.\n""vmaCreateImage() returned %s", lut::to_string(res).c_str());
		}

		lut::Image colorImage(aAllocator.allocator, image, allocation);
		// Create the image view
		VkImageViewCreateInfo viewInfo{};
		viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		viewInfo.image = colorImage.image;
		viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
		viewInfo.format = aWindow.swapchainFormat;
		viewInfo.components = VkComponentMapping{};
		viewInfo.subresourceRange = VkImageSubresourceRange{
			VK_IMAGE_ASPECT_COLOR_BIT,
			0, 1,
			0, 1
		};
		VkImageView view = VK_NULL_HANDLE;
		if (auto const res = vkCreateImageView(aWindow.device, &viewInfo, nullptr, &view); VK_SUCCESS != res)
		{
			throw lut::Error("Unable to create image view\n""vkCreateImageView() returned %s", lut::to_string(res).c_str());
		}
		return{ std::move(colorImage), lut::ImageView(aWindow.device, view) };
	}

	glm::vec4 update_user_state(UserState& aState, float aElapsedTime)
	{
		auto& cam = aState.camera2world;

		if (aState.inputMap[std::size_t(EInputState::mousing)])
		{
			if (aState.wasMousing)
			{
				auto const sens = cfg::kCameraMouseSensitivity;
				auto const dx = sens * (aState.mouseX - aState.previousX);
				auto const dy = sens * (aState.mouseY - aState.previousY);

				cam = cam * glm::rotate(-dy, glm::vec3(1.f, 0.f, 0.f));
				cam = cam * glm::rotate(-dx, glm::vec3(0.f, 1.f, 0.f));
			}

			aState.previousX = aState.mouseX;
			aState.previousY = aState.mouseY;
			aState.wasMousing = true;
		}
		else
		{
			aState.wasMousing = false;
		}
		auto const move = aElapsedTime * cfg::kCameraBaseSpeed * (aState.inputMap[std::size_t(EInputState::fast)] ? cfg::kCameraFastMult : 1.f) *
			(aState.inputMap[std::size_t(EInputState::slow)] ? cfg::kCameraSlowMult : 1.f);

		if (aState.inputMap[std::size_t(EInputState::forward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, -move));
		if (aState.inputMap[std::size_t(EInputState::backward)])
			cam = cam * glm::translate(glm::vec3(0.f, 0.f, +move));

		if (aState.inputMap[std::size_t(EInputState::strafeLeft)])
			cam = cam * glm::translate(glm::vec3(-move, 0.f, 0.f));
		if (aState.inputMap[std::size_t(EInputState::strafeRight)])
			cam = cam * glm::translate(glm::vec3(+move, 0.f, 0.f));

		if (aState.inputMap[std::size_t(EInputState::levitate)])
			cam = cam * glm::translate(glm::vec3(0.f, +move, 0.f));
		if (aState.inputMap[std::size_t(EInputState::sink)])
			cam = cam * glm::translate(glm::vec3(0.f, -move, 0.f));


		return glm::vec4(cam[3][0], cam[3][1], cam[3][2], 1.0f);

	}

	BakedMeshBuffer upload_mesh_buffer(lut::VulkanContext const& aContext, lut::Allocator const& aAllocator, BakedModel const& model)
	{
		std::vector<lut::Buffer> positions;
		std::vector<lut::Buffer> texcoords;
		std::vector<lut::Buffer> normals;
		std::vector<lut::Buffer> indices;

		for (std::size_t i = 0; i < model.meshes.size(); ++i)
		{
			std::vector<glm::vec3> Positions;
			std::vector<glm::vec2> TexCoords;
			std::vector<glm::vec3> Normals;
			std::vector<std::uint32_t> Indices;

			for (std::size_t j = 0; j < model.meshes[i].positions.size(); ++j)
			{
				Positions.push_back(model.meshes[i].positions[j]);
				TexCoords.push_back(model.meshes[i].texcoords[j]);
				Normals.push_back(model.meshes[i].normals[j]);
			}

			for (std::size_t j = 0; j < model.meshes[i].indices.size(); ++j)
			{
				Indices.push_back(model.meshes[i].indices[j]);
			}

			lut::Buffer vertexPosGPU = lut::create_buffer(
				aAllocator,
				sizeof(glm::vec3) * Positions.size(),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VMA_MEMORY_USAGE_GPU_ONLY
			);

			lut::Buffer vertexTexGPU = lut::create_buffer(
				aAllocator,
				sizeof(glm::vec2) * TexCoords.size(),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VMA_MEMORY_USAGE_GPU_ONLY
			);

			lut::Buffer vertexNormalGPU = lut::create_buffer(
				aAllocator,
				sizeof(glm::vec3) * Normals.size(),
				VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VMA_MEMORY_USAGE_GPU_ONLY
			);

			lut::Buffer indiceGPU = lut::create_buffer(
				aAllocator,
				sizeof(std::uint32_t) * Indices.size(),
				VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VMA_MEMORY_USAGE_GPU_ONLY
			);

			lut::Buffer posStaging = lut::create_buffer(
				aAllocator,
				sizeof(glm::vec3) * Positions.size(),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_MEMORY_USAGE_CPU_TO_GPU
			);
			lut::Buffer TexStaging = lut::create_buffer(
				aAllocator,
				sizeof(glm::vec2) * TexCoords.size(),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_MEMORY_USAGE_CPU_TO_GPU
			);

			lut::Buffer normalStaging = lut::create_buffer(
				aAllocator,
				sizeof(glm::vec3) * Normals.size(),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_MEMORY_USAGE_CPU_TO_GPU
			);

			lut::Buffer indiceStaging = lut::create_buffer(
				aAllocator,
				sizeof(std::uint32_t) * Indices.size(),
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
				VMA_MEMORY_USAGE_CPU_TO_GPU
			);

			void* posPtr = nullptr;
			if (auto const res = vmaMapMemory(aAllocator.allocator, posStaging.allocation, &posPtr); VK_SUCCESS != res)
			{
				throw lut::Error("Mapping memory for writing\n""vmaMapMemory() returned %s", lut::to_string(res).c_str());
			}
			std::memcpy(posPtr, Positions.data(), sizeof(glm::vec3) * Positions.size());
			vmaUnmapMemory(aAllocator.allocator, posStaging.allocation);

			void* texPtr = nullptr;
			if (auto const res = vmaMapMemory(aAllocator.allocator, TexStaging.allocation, &texPtr); VK_SUCCESS != res)
			{
				throw lut::Error("Mapping memory for writing\n""vmaMapMemory() returned %s", lut::to_string(res).c_str());
			}
			std::memcpy(texPtr, TexCoords.data(), sizeof(glm::vec2) * TexCoords.size());
			vmaUnmapMemory(aAllocator.allocator, TexStaging.allocation);

			void* normalPtr = nullptr;
			if (auto const res = vmaMapMemory(aAllocator.allocator, normalStaging.allocation, &normalPtr); VK_SUCCESS != res)
			{
				throw lut::Error("Mapping memory for writing\n""vmaMapMemory() returned %s", lut::to_string(res).c_str());
			}
			std::memcpy(normalPtr, Normals.data(), sizeof(glm::vec3) * Normals.size());
			vmaUnmapMemory(aAllocator.allocator, normalStaging.allocation);

			void* indicePtr = nullptr;
			if (auto const res = vmaMapMemory(aAllocator.allocator, indiceStaging.allocation, &indicePtr); VK_SUCCESS != res)
			{
				throw lut::Error("Mapping memory for writing\n""vmaMapMemory() returned %s", lut::to_string(res).c_str());
			}
			std::memcpy(indicePtr, Indices.data(), sizeof(std::uint32_t) * Indices.size());
			vmaUnmapMemory(aAllocator.allocator, indiceStaging.allocation);

			lut::Fence uploadComplete = create_fence(aContext);
			// Queue data uploads from staging buffers to the final buffers
			// This uses a separate command pool for simplicity.
			lut::CommandPool uploadPool = create_command_pool(aContext);
			VkCommandBuffer uploadCmd = alloc_command_buffer(aContext, uploadPool.handle);

			VkCommandBufferBeginInfo beginInfo{};
			beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
			beginInfo.flags = 0;
			beginInfo.pInheritanceInfo = nullptr;

			if (auto const res = vkBeginCommandBuffer(uploadCmd, &beginInfo); VK_SUCCESS != res)
			{
				throw lut::Error("Beginning command buffer recording\n""vkBeginCommandBuffer() returned %s", lut::to_string(res).c_str());
			}
			VkBufferCopy pcopy{};
			pcopy.size = sizeof(glm::vec3) * Positions.size();

			vkCmdCopyBuffer(uploadCmd, posStaging.buffer, vertexPosGPU.buffer, 1, &pcopy);

			lut::buffer_barrier(uploadCmd,
				vertexPosGPU.buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
			);

			VkBufferCopy tcopy{};
			tcopy.size = sizeof(glm::vec2) * TexCoords.size();

			vkCmdCopyBuffer(uploadCmd, TexStaging.buffer, vertexTexGPU.buffer, 1, &tcopy);

			lut::buffer_barrier(uploadCmd,
				vertexTexGPU.buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
			);

			VkBufferCopy ncopy{};
			ncopy.size = sizeof(glm::vec3) * Normals.size();

			vkCmdCopyBuffer(uploadCmd, normalStaging.buffer, vertexNormalGPU.buffer, 1, &ncopy);

			lut::buffer_barrier(uploadCmd,
				vertexNormalGPU.buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
			);

			VkBufferCopy icopy{};
			icopy.size = sizeof(std::uint32_t) * Indices.size();

			vkCmdCopyBuffer(uploadCmd, indiceStaging.buffer, indiceGPU.buffer, 1, &icopy);

			lut::buffer_barrier(uploadCmd,
				indiceGPU.buffer,
				VK_ACCESS_TRANSFER_WRITE_BIT,
				VK_ACCESS_VERTEX_ATTRIBUTE_READ_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_VERTEX_INPUT_BIT
			);

			if (auto const res = vkEndCommandBuffer(uploadCmd); VK_SUCCESS != res)
			{
				throw lut::Error("Ending command buffer recording\n""vkEndCommandBuffer() returned %s", lut::to_string(res).c_str());
			}

			// Submit transfer commands
			VkSubmitInfo submitInfo{};
			submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			submitInfo.commandBufferCount = 1;
			submitInfo.pCommandBuffers = &uploadCmd;

			if (auto const res = vkQueueSubmit(aContext.graphicsQueue, 1, &submitInfo, uploadComplete.handle); VK_SUCCESS != res)
			{
				throw lut::Error("Submitting commands\n""vkQueueSubmit() returned %s", lut::to_string(res).c_str());
			}

			if (auto const res = vkWaitForFences(aContext.device, 1, &uploadComplete.handle, VK_TRUE, std::numeric_limits<std::uint64_t>::max()); VK_SUCCESS != res)
			{
				throw lut::Error("Waiting for upload to complete\n""vkWaitForFences() returned %s", lut::to_string(res).c_str());
			}

			positions.push_back(std::move(vertexPosGPU));
			texcoords.push_back(std::move(vertexTexGPU));
			normals.push_back(std::move(vertexNormalGPU));
			indices.push_back(std::move(indiceGPU));
		}

		return BakedMeshBuffer{
			std::move(positions),
			std::move(texcoords),
			std::move(normals),
			std::move(indices)
		};
	}

	void updateDescriptorSet(lut::VulkanWindow const& aWindow, VkDescriptorSet const& Descriptor, VkImageView const& aImageView,
		VkSampler const& Sampler)
	{
		{
			VkWriteDescriptorSet desc[1]{};

			VkDescriptorImageInfo textureInfo{};
			textureInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
			textureInfo.imageView = aImageView;
			textureInfo.sampler = Sampler;

			desc[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			desc[0].dstSet = Descriptor;
			desc[0].dstBinding = 0;
			desc[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
			desc[0].descriptorCount = 1;
			desc[0].pImageInfo = &textureInfo;

			constexpr auto numSets = sizeof(desc) / sizeof(desc[0]);
			vkUpdateDescriptorSets(aWindow.device, numSets, desc, 0, nullptr);
		}
	}
}


//EOF vim:syntax=cpp:foldmethod=marker:ts=4:noexpandtab: 
