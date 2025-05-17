#ifdef _WIN64
#include <Windows.h>
#else
#include <pthread.h>
#endif
#include <map>
#include <stdio.h>
#include <cstring>  // For memset

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

const char *KernelSource = "\n" \
"__kernel void GaussianBlurKernel(                                        \n" \
"   int p_Width,                                                          \n" \
"   int p_Height,                                                         \n" \
"   float p_Radius,                                                       \n" \
"   int p_Quality,                                                        \n" \
"   float p_MaskStrength,                                                 \n" \
"   __read_only image2d_t p_Input,                                        \n" \
"   __read_only image2d_t p_Mask,                                         \n" \
"   __global float* p_Output)                                             \n" \
"{                                                                        \n" \
"   const int x = get_global_id(0);                                       \n" \
"   const int y = get_global_id(1);                                       \n" \
"                                                                         \n" \
"   if ((x < p_Width) && (y < p_Height))                                  \n" \
"   {                                                                     \n" \
"       // Set up sampler for reading images                              \n" \
"       const sampler_t sampler = CLK_NORMALIZED_COORDS_TRUE |            \n" \
"                                CLK_ADDRESS_CLAMP_TO_EDGE |              \n" \
"                                CLK_FILTER_LINEAR;                        \n" \
"                                                                         \n" \
"       // Read mask value if available                                   \n" \
"       float maskValue = 1.0f;                                           \n" \
"       if (p_MaskStrength > 0.0f) {                                      \n" \
"           float2 uv = (float2)((float)x / p_Width, (float)y / p_Height);\n" \
"           float4 maskColor = read_imagef(p_Mask, sampler, uv);          \n" \
"           // Use alpha channel for mask                                 \n" \
"           maskValue = maskColor.w;                                      \n" \
"           // Apply mask strength                                        \n" \
"           maskValue = 1.0f - (p_MaskStrength * maskValue);              \n" \
"       }                                                                 \n" \
"                                                                         \n" \
"       // Calculate effective blur radius based on mask                  \n" \
"       float effectiveRadius = p_Radius * maskValue;                     \n" \
"                                                                         \n" \
"       const int index = ((y * p_Width) + x) * 4;                        \n" \
"                                                                         \n" \
"       // Early exit if no blur needed                                   \n" \
"       if (effectiveRadius <= 0.0f) {                                    \n" \
"           float2 uv = (float2)((float)x / p_Width, (float)y / p_Height);\n" \
"           float4 srcColor = read_imagef(p_Input, sampler, uv);          \n" \
"           p_Output[index + 0] = srcColor.x;                             \n" \
"           p_Output[index + 1] = srcColor.y;                             \n" \
"           p_Output[index + 2] = srcColor.z;                             \n" \
"           p_Output[index + 3] = srcColor.w;                             \n" \
"           return;                                                       \n" \
"       }                                                                 \n" \
"                                                                         \n" \
"       // Gaussian blur implementation                                   \n" \
"       float4 sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);                    \n" \
"       float weightSum = 0.0f;                                           \n" \
"                                                                         \n" \
"       // Sample in a circle pattern                                     \n" \
"       for (int i = 0; i < p_Quality; ++i) {                             \n" \
"           float angle = (2.0f * 3.14159f * i) / (float)p_Quality;       \n" \
"                                                                         \n" \
"           for (float distance = 1.0f;                                   \n" \
"                distance <= effectiveRadius;                             \n" \
"                distance += 1.0f) {                                      \n" \
"               float sampleX = x + cos(angle) * distance;                \n" \
"               float sampleY = y + sin(angle) * distance;                \n" \
"                                                                         \n" \
"               // Convert to normalized texture coordinates              \n" \
"               float2 uv = (float2)(sampleX / p_Width, sampleY / p_Height);\n" \
"                                                                         \n" \
"               // Sample using texture                                   \n" \
"               float4 color = read_imagef(p_Input, sampler, uv);         \n" \
"                                                                         \n" \
"               // Calculate Gaussian weight                              \n" \
"               float weight = exp(-(distance * distance) /               \n" \
"                               (2.0f * effectiveRadius * effectiveRadius));\n" \
"                                                                         \n" \
"               // Accumulate weighted color                              \n" \
"               sum += color * weight;                                    \n" \
"               weightSum += weight;                                      \n" \
"           }                                                             \n" \
"       }                                                                 \n" \
"                                                                         \n" \
"       // Add center pixel with highest weight                           \n" \
"       float2 centerUV = (float2)((float)x / p_Width, (float)y / p_Height);\n" \
"       float4 centerColor = read_imagef(p_Input, sampler, centerUV);     \n" \
"       float centerWeight = 1.0f;                                        \n" \
"       sum += centerColor * centerWeight;                                \n" \
"       weightSum += centerWeight;                                        \n" \
"                                                                         \n" \
"       // Normalize by total weight                                      \n" \
"       if (weightSum > 0.0f) {                                           \n" \
"           sum /= weightSum;                                             \n" \
"       }                                                                 \n" \
"                                                                         \n" \
"       // Write to output                                                \n" \
"       p_Output[index + 0] = sum.x;                                      \n" \
"       p_Output[index + 1] = sum.y;                                      \n" \
"       p_Output[index + 2] = sum.z;                                      \n" \
"       p_Output[index + 3] = sum.w;                                      \n" \
"   }                                                                     \n" \
"}                                                                        \n" \
"\n";

void CheckError(cl_int p_Error, const char* p_Msg)
{
    if (p_Error != CL_SUCCESS)
    {
        fprintf(stderr, "%s [%d]\n", p_Msg, p_Error);
    }
}

class Locker
{
public:
    Locker()
    {
#ifdef _WIN64
        InitializeCriticalSection(&mutex);
#else
        pthread_mutex_init(&mutex, NULL);
#endif
    }

    ~Locker()
    {
#ifdef _WIN64
        DeleteCriticalSection(&mutex);
#else
        pthread_mutex_destroy(&mutex);
#endif
    }

    void Lock()
    {
#ifdef _WIN64
        EnterCriticalSection(&mutex);
#else
        pthread_mutex_lock(&mutex);
#endif
    }

    void Unlock()
    {
#ifdef _WIN64
        LeaveCriticalSection(&mutex);
#else
        pthread_mutex_unlock(&mutex);
#endif
    }

private:
#ifdef _WIN64
    CRITICAL_SECTION mutex;
#else
    pthread_mutex_t mutex;
#endif
};

void RunOpenCLKernel(void* p_CmdQ, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength,
                     const float* p_Input, const float* p_Mask, float* p_Output)
{
    cl_int error;

    cl_command_queue cmdQ = static_cast<cl_command_queue>(p_CmdQ);

    // store device id and kernel per command queue (required for multi-GPU systems)
    static std::map<cl_command_queue, cl_device_id> deviceIdMap;
    static std::map<cl_command_queue, cl_kernel> kernelMap;
    static std::map<cl_command_queue, cl_context> contextMap;
    static std::map<cl_command_queue, cl_program> programMap;

    static Locker locker; // simple lock to control access to the above maps from multiple threads

    locker.Lock();

    // find the device id corresponding to the command queue
    cl_device_id deviceId = NULL;
    cl_context clContext = NULL;
    
    if (deviceIdMap.find(cmdQ) == deviceIdMap.end())
    {
        error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_DEVICE, sizeof(cl_device_id), &deviceId, NULL);
        CheckError(error, "Unable to get the device");
        deviceIdMap[cmdQ] = deviceId;
        
        error = clGetCommandQueueInfo(cmdQ, CL_QUEUE_CONTEXT, sizeof(cl_context), &clContext, NULL);
        CheckError(error, "Unable to get the context");
        contextMap[cmdQ] = clContext;
    }
    else
    {
        deviceId = deviceIdMap[cmdQ];
        clContext = contextMap[cmdQ];
    }

    // find the program kernel corresponding to the command queue
    cl_kernel kernel = NULL;
    cl_program program = NULL;
    
    if (kernelMap.find(cmdQ) == kernelMap.end())
    {
        program = clCreateProgramWithSource(clContext, 1, (const char **)&KernelSource, NULL, &error);
        CheckError(error, "Unable to create program");
        programMap[cmdQ] = program;

        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        CheckError(error, "Unable to build program");

        kernel = clCreateKernel(program, "GaussianBlurKernel", &error);
        CheckError(error, "Unable to create kernel");
        kernelMap[cmdQ] = kernel;
    }
    else
    {
        kernel = kernelMap[cmdQ];
        program = programMap[cmdQ];
    }

    locker.Unlock();

    // Create image format description
    cl_image_format format;
    format.image_channel_order = CL_RGBA;
    format.image_channel_data_type = CL_FLOAT;

    // Create image description
    cl_image_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.image_type = CL_MEM_OBJECT_IMAGE2D;
    desc.image_width = p_Width;
    desc.image_height = p_Height;
    desc.image_row_pitch = 0; // Let OpenCL calculate
    desc.image_array_size = 1;
    
    // Create input image
    cl_mem inputImage = clCreateImage(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                     &format, &desc, (void*)p_Input, &error);
    CheckError(error, "Unable to create input image");
    
    // Create mask image (if available)
    cl_mem maskImage = NULL;
    if (p_Mask) {
        maskImage = clCreateImage(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                 &format, &desc, (void*)p_Mask, &error);
        CheckError(error, "Unable to create mask image");
    } else {
        // Create a dummy mask if none provided
        float* dummyMask = new float[p_Width * p_Height * 4]();
        maskImage = clCreateImage(clContext, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                 &format, &desc, dummyMask, &error);
        CheckError(error, "Unable to create dummy mask image");
        delete[] dummyMask;
    }

    // Create output buffer
    cl_mem outputBuffer = clCreateBuffer(clContext, CL_MEM_WRITE_ONLY, 
                                        p_Width * p_Height * 4 * sizeof(float), NULL, &error);
    CheckError(error, "Unable to create output buffer");

    // Set kernel arguments
    int count = 0;
    error  = clSetKernelArg(kernel, count++, sizeof(int), &p_Width);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Height);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_Radius);
    error |= clSetKernelArg(kernel, count++, sizeof(int), &p_Quality);
    error |= clSetKernelArg(kernel, count++, sizeof(float), &p_MaskStrength);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &inputImage);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &maskImage);
    error |= clSetKernelArg(kernel, count++, sizeof(cl_mem), &outputBuffer);
    CheckError(error, "Unable to set kernel arguments");

    // Determine work group size
    size_t localWorkSize[2];
    error = clGetKernelWorkGroupInfo(kernel, deviceId, CL_KERNEL_WORK_GROUP_SIZE, 
                                    sizeof(size_t), &localWorkSize[0], NULL);
    CheckError(error, "Unable to get kernel work group info");
    
    localWorkSize[0] = 16;  // Adjust as needed
    localWorkSize[1] = 16;  // Adjust as needed
    
    // Calculate global work size
    size_t globalWorkSize[2];
    globalWorkSize[0] = ((p_Width + localWorkSize[0] - 1) / localWorkSize[0]) * localWorkSize[0];
    globalWorkSize[1] = ((p_Height + localWorkSize[1] - 1) / localWorkSize[1]) * localWorkSize[1];

    // Execute kernel
    error = clEnqueueNDRangeKernel(cmdQ, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
    CheckError(error, "Unable to enqueue kernel");
    
    // Read back the results
    error = clEnqueueReadBuffer(cmdQ, outputBuffer, CL_TRUE, 0, 
                               p_Width * p_Height * 4 * sizeof(float), p_Output, 0, NULL, NULL);
    CheckError(error, "Unable to read output buffer");
    
    // Clean up resources
    clReleaseMemObject(inputImage);
    clReleaseMemObject(maskImage);
    clReleaseMemObject(outputBuffer);
}