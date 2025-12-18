#pragma once

#include <vulkan/vulkan_core.h>

#if defined(VK_EXT_swapchain_maintenance1) && !defined(VK_KHR_swapchain_maintenance1)
typedef VkReleaseSwapchainImagesInfoEXT VkReleaseSwapchainImagesInfoKHR;
#define VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_KHR \
	VK_STRUCTURE_TYPE_RELEASE_SWAPCHAIN_IMAGES_INFO_EXT
#endif