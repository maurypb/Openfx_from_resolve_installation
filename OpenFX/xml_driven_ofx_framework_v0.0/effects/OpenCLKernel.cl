// Simplified OpenCLKernel.cpp - Contains ONLY kernel source and bridge function
// All setup/teardown moved to KernelWrappers.cpp

#ifndef __APPLE__
#include <CL/cl.h>
#else
#include <OpenCL/cl.h>
#endif

// Pure OpenCL kernel source - this is what effect authors write
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
"           maskValue = maskColor.w;                                      \n" \
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
"               float2 uv = (float2)(sampleX / p_Width, sampleY / p_Height);\n" \
"                                                                         \n" \
"               float4 color = read_imagef(p_Input, sampler, uv);         \n" \
"                                                                         \n" \
"               float weight = exp(-(distance * distance) /               \n" \
"                               (2.0f * effectiveRadius * effectiveRadius));\n" \
"                                                                         \n" \
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
"       if (weightSum > 0.0f) {                                           \n" \
"           sum /= weightSum;                                             \n" \
"       }                                                                 \n" \
"                                                                         \n" \
"       p_Output[index + 0] = sum.x;                                      \n" \
"       p_Output[index + 1] = sum.y;                                      \n" \
"       p_Output[index + 2] = sum.z;                                      \n" \
"       p_Output[index + 3] = sum.w;                                      \n" \
"   }                                                                     \n" \
"}                                                                        \n" \
"\n";

// Simple bridge function to provide access to kernel source
// This is the only "plumbing" left in the kernel file
extern "C" const char* get_opencl_kernel_source() {
    return KernelSource;
}

// Simple bridge function to get kernel name
extern "C" const char* get_opencl_kernel_name() {
    return "GaussianBlurKernel";
}