// Simplified MetalKernel.mm - Contains ONLY kernel source and bridge function
// All setup/teardown moved to KernelWrappers.cpp

#ifdef __APPLE__
#import <Metal/Metal.h>
#endif

// Pure Metal kernel source - this is what effect authors write
const char* kernelSource = \
"#include <metal_stdlib>\n" \
"using namespace metal; \n" \
"kernel void GaussianBlurKernel(constant int& p_Width [[buffer (11)]], constant int& p_Height [[buffer (12)]], constant float& p_Radius [[buffer (13)]],    \n" \
"                             constant int& p_Quality [[buffer (14)]], constant float& p_MaskStrength [[buffer (15)]],                                     \n" \
"                             texture2d<float, access::sample> p_Input [[texture(0)]], texture2d<float, access::sample> p_Mask [[texture(1)]],             \n" \
"                             device float* p_Output [[buffer (8)]], uint2 id [[ thread_position_in_grid ]])                                               \n" \
"{                                                                                                                                                         \n" \
"   if ((id.x < p_Width) && (id.y < p_Height))                                                                                                             \n" \
"   {                                                                                                                                                      \n" \
"       constexpr sampler textureSampler(coord::normalized, address::clamp_to_edge, filter::linear);                                                       \n" \
"                                                                                                                                                          \n" \
"       // Read mask value if available                                                                                                                    \n" \
"       float maskValue = 1.0f;                                                                                                                            \n" \
"       if (p_MaskStrength > 0.0f) {                                                                                                                       \n" \
"           float2 uv = float2(float(id.x) / float(p_Width), float(id.y) / float(p_Height));                                                               \n" \
"           float4 maskColor = p_Mask.sample(textureSampler, uv);                                                                                          \n" \
"           maskValue = maskColor.a;                                                                                                                       \n" \
"           maskValue = 1.0f - (p_MaskStrength * maskValue);                                                                                               \n" \
"       }                                                                                                                                                  \n" \
"                                                                                                                                                          \n" \
"       // Calculate effective blur radius based on mask                                                                                                   \n" \
"       float effectiveRadius = p_Radius * maskValue;                                                                                                      \n" \
"                                                                                                                                                          \n" \
"       const int index = ((id.y * p_Width) + id.x) * 4;                                                                                                   \n" \
"                                                                                                                                                          \n" \
"       // Early exit if no blur needed                                                                                                                    \n" \
"       if (effectiveRadius <= 0.0f) {                                                                                                                     \n" \
"           float2 uv = float2(float(id.x) / float(p_Width), float(id.y) / float(p_Height));                                                               \n" \
"           float4 srcColor = p_Input.sample(textureSampler, uv);                                                                                          \n" \
"           p_Output[index + 0] = srcColor.r;                                                                                                              \n" \
"           p_Output[index + 1] = srcColor.g;                                                                                                              \n" \
"           p_Output[index + 2] = srcColor.b;                                                                                                              \n" \
"           p_Output[index + 3] = srcColor.a;                                                                                                              \n" \
"           return;                                                                                                                                        \n" \
"       }                                                                                                                                                  \n" \
"                                                                                                                                                          \n" \
"       // Gaussian blur implementation                                                                                                                    \n" \
"       float4 sum = float4(0.0f);                                                                                                                         \n" \
"       float weightSum = 0.0f;                                                                                                                            \n" \
"                                                                                                                                                          \n" \
"       // Sample in a circle pattern                                                                                                                      \n" \
"       for (int i = 0; i < p_Quality; ++i) {                                                                                                              \n" \
"           float angle = (2.0f * 3.14159f * i) / float(p_Quality);                                                                                        \n" \
"                                                                                                                                                          \n" \
"           for (float distance = 1.0f; distance <= effectiveRadius; distance += 1.0f) {                                                                   \n" \
"               float sampleX = float(id.x) + cos(angle) * distance;                                                                                       \n" \
"               float sampleY = float(id.y) + sin(angle) * distance;                                                                                       \n" \
"                                                                                                                                                          \n" \
"               float2 uv = float2(sampleX / float(p_Width), sampleY / float(p_Height));                                                                   \n" \
"                                                                                                                                                          \n" \
"               float4 color = p_Input.sample(textureSampler, uv);                                                                                         \n" \
"                                                                                                                                                          \n" \
"               float weight = exp(-(distance * distance) / (2.0f * effectiveRadius * effectiveRadius));                                                   \n" \
"                                                                                                                                                          \n" \
"               sum += color * weight;                                                                                                                     \n" \
"               weightSum += weight;                                                                                                                       \n" \
"           }                                                                                                                                              \n" \
"       }                                                                                                                                                  \n" \
"                                                                                                                                                          \n" \
"       // Add center pixel with highest weight                                                                                                            \n" \
"       float2 centerUV = float2(float(id.x) / float(p_Width), float(id.y) / float(p_Height));                                                             \n" \
"       float4 centerColor = p_Input.sample(textureSampler, centerUV);                                                                                     \n" \
"       float centerWeight = 1.0f;                                                                                                                         \n" \
"       sum += centerColor * centerWeight;                                                                                                                 \n" \
"       weightSum += centerWeight;                                                                                                                         \n" \
"                                                                                                                                                          \n" \
"       if (weightSum > 0.0f) {                                                                                                                            \n" \
"           sum /= weightSum;                                                                                                                              \n" \
"       }                                                                                                                                                  \n" \
"                                                                                                                                                          \n" \
"       p_Output[index + 0] = sum.r;                                                                                                                       \n" \
"       p_Output[index + 1] = sum.g;                                                                                                                       \n" \
"       p_Output[index + 2] = sum.b;                                                                                                                       \n" \
"       p_Output[index + 3] = sum.a;                                                                                                                       \n" \
"   }                                                                                                                                                      \n" \
"}                                                                                                                                                         \n";

// Simple bridge functions to provide access to kernel source
// This is the only "plumbing" left in the kernel file
extern "C" const char* get_metal_kernel_source() {
    return kernelSource;
}

extern "C" const char* get_metal_kernel_name() {
    return "GaussianBlurKernel";
}