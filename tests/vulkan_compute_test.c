#include <vulkan/vulkan.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define ELEMENT_COUNT 1024
#define WORKGROUP_SIZE 256
#define BAIL(msg) do { fprintf(stderr, "FAIL: %s (VkResult=%d)\n", msg, res); exit(1); } while(0)

static uint32_t *read_spirv(const char *path, size_t *pSize) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END);
    *pSize = (size_t)ftell(f);
    rewind(f);
    uint32_t *buf = malloc(*pSize);
    fread(buf, 1, *pSize, f);
    fclose(f);
    return buf;
}

static uint32_t find_memory_type(VkPhysicalDevice phys, uint32_t typeBits, VkMemoryPropertyFlags props) {
    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(phys, &mem);
    for (uint32_t i = 0; i < mem.memoryTypeCount; i++) {
        if ((typeBits & (1u << i)) && (mem.memoryTypes[i].propertyFlags & props) == props)
            return i;
    }
    fprintf(stderr, "FAIL: no suitable memory type\n");
    exit(1);
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    VkResult res;

    /* 1. Instance */
    VkApplicationInfo appInfo = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "vulkan_compute_test",
        .apiVersion = VK_API_VERSION_1_0,
    };
    VkInstanceCreateInfo instCI = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &appInfo,
    };
    VkInstance instance;
    res = vkCreateInstance(&instCI, NULL, &instance);
    if (res != VK_SUCCESS) BAIL("vkCreateInstance");
    printf("[OK] Vulkan instance created\n");

    /* 2. Physical device */
    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, NULL);
    if (devCount == 0) { fprintf(stderr, "FAIL: no Vulkan devices\n"); exit(1); }

    VkPhysicalDevice *devs = malloc(devCount * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(instance, &devCount, devs);

    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkPhysicalDeviceProperties chosenProps;

    printf("[..] Found %u physical device(s):\n", devCount);
    for (uint32_t i = 0; i < devCount; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devs[i], &props);
        printf("     [%u] %s (type=%d)\n", i, props.deviceName, props.deviceType);
        if (strstr(props.deviceName, "Adreno") ||
            strstr(props.deviceName, "Direct3D12") ||
            strstr(props.deviceName, "D3D12") ||
            strstr(props.deviceName, "Dozen")) {
            phys = devs[i];
            chosenProps = props;
        }
    }
    if (phys == VK_NULL_HANDLE) {
        phys = devs[0];
        vkGetPhysicalDeviceProperties(phys, &chosenProps);
        printf("[!!] No Adreno/D3D12 device found, falling back to: %s\n", chosenProps.deviceName);
    }
    free(devs);
    printf("[OK] Selected device: %s\n", chosenProps.deviceName);

    /* 3. Queue family (compute) */
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, NULL);
    VkQueueFamilyProperties *qfProps = malloc(qfCount * sizeof(VkQueueFamilyProperties));
    vkGetPhysicalDeviceQueueFamilyProperties(phys, &qfCount, qfProps);

    uint32_t computeQF = UINT32_MAX;
    for (uint32_t i = 0; i < qfCount; i++) {
        if (qfProps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { computeQF = i; break; }
    }
    free(qfProps);
    if (computeQF == UINT32_MAX) { fprintf(stderr, "FAIL: no compute queue\n"); exit(1); }
    printf("[OK] Compute queue family: %u\n", computeQF);

    /* 4. Logical device */
    float qPri = 1.0f;
    VkDeviceQueueCreateInfo qCI = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = computeQF, .queueCount = 1, .pQueuePriorities = &qPri,
    };
    VkDeviceCreateInfo devCI = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1, .pQueueCreateInfos = &qCI,
    };
    VkDevice device;
    res = vkCreateDevice(phys, &devCI, NULL, &device);
    if (res != VK_SUCCESS) BAIL("vkCreateDevice");
    VkQueue queue;
    vkGetDeviceQueue(device, computeQF, 0, &queue);
    printf("[OK] Logical device and queue created\n");

    /* 5. Buffer (host-visible) */
    VkDeviceSize bufSize = ELEMENT_COUNT * sizeof(uint32_t);
    VkBufferCreateInfo bufCI = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size = bufSize, .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    VkBuffer buffer;
    res = vkCreateBuffer(device, &bufCI, NULL, &buffer);
    if (res != VK_SUCCESS) BAIL("vkCreateBuffer");

    VkMemoryRequirements memReq;
    vkGetBufferMemoryRequirements(device, buffer, &memReq);
    VkMemoryAllocateInfo allocInfo = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .allocationSize = memReq.size,
        .memoryTypeIndex = find_memory_type(phys, memReq.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT),
    };
    VkDeviceMemory memory;
    res = vkAllocateMemory(device, &allocInfo, NULL, &memory);
    if (res != VK_SUCCESS) BAIL("vkAllocateMemory");
    vkBindBufferMemory(device, buffer, memory, 0);

    uint32_t *mapped;
    vkMapMemory(device, memory, 0, bufSize, 0, (void **)&mapped);
    for (uint32_t i = 0; i < ELEMENT_COUNT; i++) mapped[i] = i + 1;
    vkUnmapMemory(device, memory);
    printf("[OK] Buffer filled with test data (1..%d)\n", ELEMENT_COUNT);

    /* 6. Descriptor set layout + pool + set */
    VkDescriptorSetLayoutBinding binding = {
        .binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetLayoutCreateInfo dslCI = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 1, .pBindings = &binding,
    };
    VkDescriptorSetLayout dsl;
    res = vkCreateDescriptorSetLayout(device, &dslCI, NULL, &dsl);
    if (res != VK_SUCCESS) BAIL("vkCreateDescriptorSetLayout");

    VkDescriptorPoolSize poolSize = { .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = 1 };
    VkDescriptorPoolCreateInfo dpCI = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = 1, .poolSizeCount = 1, .pPoolSizes = &poolSize,
    };
    VkDescriptorPool dPool;
    res = vkCreateDescriptorPool(device, &dpCI, NULL, &dPool);
    if (res != VK_SUCCESS) BAIL("vkCreateDescriptorPool");

    VkDescriptorSetAllocateInfo dsAI = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = dPool, .descriptorSetCount = 1, .pSetLayouts = &dsl,
    };
    VkDescriptorSet ds;
    res = vkAllocateDescriptorSets(device, &dsAI, &ds);
    if (res != VK_SUCCESS) BAIL("vkAllocateDescriptorSets");

    VkDescriptorBufferInfo dbi = { .buffer = buffer, .offset = 0, .range = bufSize };
    VkWriteDescriptorSet wds = {
        .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
        .dstSet = ds, .dstBinding = 0, .descriptorCount = 1,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .pBufferInfo = &dbi,
    };
    vkUpdateDescriptorSets(device, 1, &wds, 0, NULL);

    /* 7. Shader module */
    size_t spirvSize;
    uint32_t *spirvCode = read_spirv("multiply.spv", &spirvSize);
    VkShaderModuleCreateInfo smCI = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = spirvSize, .pCode = spirvCode,
    };
    VkShaderModule shaderMod;
    res = vkCreateShaderModule(device, &smCI, NULL, &shaderMod);
    if (res != VK_SUCCESS) BAIL("vkCreateShaderModule");
    free(spirvCode);
    printf("[OK] Compute shader loaded\n");

    /* 8. Pipeline */
    VkPipelineLayoutCreateInfo plCI = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1, .pSetLayouts = &dsl,
    };
    VkPipelineLayout pipeLayout;
    res = vkCreatePipelineLayout(device, &plCI, NULL, &pipeLayout);
    if (res != VK_SUCCESS) BAIL("vkCreatePipelineLayout");

    VkComputePipelineCreateInfo cpCI = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = shaderMod, .pName = "main",
        },
        .layout = pipeLayout,
    };
    VkPipeline pipeline;
    res = vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpCI, NULL, &pipeline);
    if (res != VK_SUCCESS) BAIL("vkCreateComputePipelines");
    printf("[OK] Compute pipeline created\n");

    /* 9. Command buffer */
    VkCommandPoolCreateInfo cpoolCI = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = computeQF,
    };
    VkCommandPool cmdPool;
    res = vkCreateCommandPool(device, &cpoolCI, NULL, &cmdPool);
    if (res != VK_SUCCESS) BAIL("vkCreateCommandPool");

    VkCommandBufferAllocateInfo cbAI = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = cmdPool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1,
    };
    VkCommandBuffer cmdBuf;
    res = vkAllocateCommandBuffers(device, &cbAI, &cmdBuf);
    if (res != VK_SUCCESS) BAIL("vkAllocateCommandBuffers");

    VkCommandBufferBeginInfo beginInfo = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    vkBeginCommandBuffer(cmdBuf, &beginInfo);
    vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &ds, 0, NULL);
    vkCmdDispatch(cmdBuf, ELEMENT_COUNT / WORKGROUP_SIZE, 1, 1);

    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_HOST_READ_BIT,
    };
    vkCmdPipelineBarrier(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &barrier, 0, NULL, 0, NULL);
    vkEndCommandBuffer(cmdBuf);

    /* 10. Submit + wait */
    VkFenceCreateInfo fenceCI = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
    VkFence fence;
    res = vkCreateFence(device, &fenceCI, NULL, &fence);
    if (res != VK_SUCCESS) BAIL("vkCreateFence");

    VkSubmitInfo submitInfo = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1, .pCommandBuffers = &cmdBuf,
    };

    double t0 = now_ms();
    res = vkQueueSubmit(queue, 1, &submitInfo, fence);
    if (res != VK_SUCCESS) BAIL("vkQueueSubmit");
    res = vkWaitForFences(device, 1, &fence, VK_TRUE, 5000000000ULL);
    if (res != VK_SUCCESS) BAIL("vkWaitForFences");
    double t1 = now_ms();
    printf("[OK] Compute dispatch completed in %.3f ms\n", t1 - t0);

    /* 11. Read back and verify */
    vkMapMemory(device, memory, 0, bufSize, 0, (void **)&mapped);
    int errors = 0;
    for (uint32_t i = 0; i < ELEMENT_COUNT; i++) {
        uint32_t expected = (i + 1) * 2;
        if (mapped[i] != expected) {
            if (errors < 10) printf("[!!] Mismatch at [%u]: got %u, expected %u\n", i, mapped[i], expected);
            errors++;
        }
    }
    vkUnmapMemory(device, memory);

    if (errors == 0) {
        printf("\n===== SUCCESS =====\n");
        printf("All %d elements verified correct (2, 4, 6, ..., %d)\n", ELEMENT_COUNT, ELEMENT_COUNT * 2);
        printf("GPU: %s\n", chosenProps.deviceName);
        printf("Compute time: %.3f ms\n", t1 - t0);
    } else {
        printf("\n===== FAILURE =====\n");
        printf("%d / %d elements mismatched\n", errors, ELEMENT_COUNT);
    }

    /* Cleanup */
    vkDestroyFence(device, fence, NULL);
    vkDestroyCommandPool(device, cmdPool, NULL);
    vkDestroyPipeline(device, pipeline, NULL);
    vkDestroyPipelineLayout(device, pipeLayout, NULL);
    vkDestroyShaderModule(device, shaderMod, NULL);
    vkDestroyDescriptorPool(device, dPool, NULL);
    vkDestroyDescriptorSetLayout(device, dsl, NULL);
    vkDestroyBuffer(device, buffer, NULL);
    vkFreeMemory(device, memory, NULL);
    vkDestroyDevice(device, NULL);
    vkDestroyInstance(instance, NULL);

    return errors == 0 ? 0 : 1;
}
