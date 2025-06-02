#import <Metal/Metal.h>

#include <unordered_map>
#include <mutex>

const char* kernelSource =  \
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
"           // Use alpha channel for mask                                                                                                                  \n" \
"           maskValue = maskColor.a;                                                                                                                       \n" \
"           // Apply mask strength                                                                                                                         \n" \
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
"               // Convert to normalized texture coordinates                                                                                               \n" \
"               float2 uv = float2(sampleX / float(p_Width), sampleY / float(p_Height));                                                                   \n" \
"                                                                                                                                                          \n" \
"               // Sample using texture                                                                                                                    \n" \
"               float4 color = p_Input.sample(textureSampler, uv);                                                                                         \n" \
"                                                                                                                                                          \n" \
"               // Calculate Gaussian weight                                                                                                               \n" \
"               float weight = exp(-(distance * distance) / (2.0f * effectiveRadius * effectiveRadius));                                                   \n" \
"                                                                                                                                                          \n" \
"               // Accumulate weighted color                                                                                                               \n" \
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
"       // Normalize by total weight                                                                                                                       \n" \
"       if (weightSum > 0.0f) {                                                                                                                            \n" \
"           sum /= weightSum;                                                                                                                              \n" \
"       }                                                                                                                                                  \n" \
"                                                                                                                                                          \n" \
"       // Write to output                                                                                                                                 \n" \
"       p_Output[index + 0] = sum.r;                                                                                                                       \n" \
"       p_Output[index + 1] = sum.g;                                                                                                                       \n" \
"       p_Output[index + 2] = sum.b;                                                                                                                       \n" \
"       p_Output[index + 3] = sum.a;                                                                                                                       \n" \
"   }                                                                                                                                                      \n" \
"}                                                                                                                                                         \n";

std::mutex s_PipelineQueueMutex;
typedef std::unordered_map<id<MTLCommandQueue>, id<MTLComputePipelineState>> PipelineQueueMap;
PipelineQueueMap s_PipelineQueueMap;

void RunMetalKernel(void* p_CmdQ, int p_Width, int p_Height, float p_Radius, int p_Quality, float p_MaskStrength, 
                    const float* p_Input, const float* p_Mask, float* p_Output)
{
    const char* kernelName = "GaussianBlurKernel";

    id<MTLCommandQueue>            queue = static_cast<id<MTLCommandQueue> >(p_CmdQ);
    id<MTLDevice>                  device = queue.device;
    id<MTLLibrary>                 metalLibrary;     // Metal library
    id<MTLFunction>                kernelFunction;   // Compute kernel
    id<MTLComputePipelineState>    pipelineState;    // Metal pipeline
    NSError* err;

    std::unique_lock<std::mutex> lock(s_PipelineQueueMutex);

    const auto it = s_PipelineQueueMap.find(queue);
    if (it == s_PipelineQueueMap.end())
    {
        id<MTLLibrary>                 metalLibrary;     // Metal library
        id<MTLFunction>                kernelFunction;   // Compute kernel
        NSError* err;

        MTLCompileOptions* options = [MTLCompileOptions new];
        options.fastMathEnabled = YES;
        if (!(metalLibrary = [device newLibraryWithSource:@(kernelSource) options:options error:&err]))
        {
            fprintf(stderr, "Failed to load metal library, %s\n", err.localizedDescription.UTF8String);
            return;
        }
        [options release];
        if (!(kernelFunction = [metalLibrary newFunctionWithName:[NSString stringWithUTF8String:kernelName]/* constantValues : constantValues */]))
        {
            fprintf(stderr, "Failed to retrieve kernel\n");
            [metalLibrary release];
            return;
        }
        if (!(pipelineState = [device newComputePipelineStateWithFunction:kernelFunction error:&err]))
        {
            fprintf(stderr, "Unable to compile, %s\n", err.localizedDescription.UTF8String);
            [metalLibrary release];
            [kernelFunction release];
            return;
        }

        s_PipelineQueueMap[queue] = pipelineState;

        //Release resources
        [metalLibrary release];
        [kernelFunction release];
    }
    else
    {
        pipelineState = it->second;
    }

    // Create textures for input and mask
    MTLTextureDescriptor *textureDescriptor = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA32Float
                                                                                                 width:p_Width
                                                                                                height:p_Height
                                                                                             mipmapped:NO];
    textureDescriptor.usage = MTLTextureUsageShaderRead;
    
    // Input texture
    id<MTLTexture> inputTexture = [device newTextureWithDescriptor:textureDescriptor];
    MTLRegion region = MTLRegionMake2D(0, 0, p_Width, p_Height);
    [inputTexture replaceRegion:region mipmapLevel:0 withBytes:p_Input bytesPerRow:p_Width * 4 * sizeof(float)];
    
    // Mask texture (if available)
    id<MTLTexture> maskTexture = nil;
    if (p_Mask) {
        maskTexture = [device newTextureWithDescriptor:textureDescriptor];
        [maskTexture replaceRegion:region mipmapLevel:0 withBytes:p_Mask bytesPerRow:p_Width * 4 * sizeof(float)];
    } else {
        // Create a dummy mask texture with all zeros if no mask provided
        float* dummyMask = new float[p_Width * p_Height * 4]();
        maskTexture = [device newTextureWithDescriptor:textureDescriptor];
        [maskTexture replaceRegion:region mipmapLevel:0 withBytes:dummyMask bytesPerRow:p_Width * 4 * sizeof(float)];
        delete[] dummyMask;
    }
    
    // Output buffer
    id<MTLBuffer> dstDeviceBuf = reinterpret_cast<id<MTLBuffer> >(p_Output);

    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    commandBuffer.label = [NSString stringWithFormat:@"GaussianBlurKernel"];

    id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];
    [computeEncoder setComputePipelineState:pipelineState];

    int exeWidth = [pipelineState threadExecutionWidth];
    MTLSize threadGroupCount = MTLSizeMake(exeWidth, 1, 1);
    MTLSize threadGroups     = MTLSizeMake((p_Width + exeWidth - 1)/exeWidth, p_Height, 1);

    // Set textures
    [computeEncoder setTexture:inputTexture atIndex:0];
    [computeEncoder setTexture:maskTexture atIndex:1];
    
    // Set output buffer and parameters
    [computeEncoder setBuffer:dstDeviceBuf offset: 0 atIndex: 8];
    [computeEncoder setBytes:&p_Width length:sizeof(int) atIndex:11];
    [computeEncoder setBytes:&p_Height length:sizeof(int) atIndex:12];
    [computeEncoder setBytes:&p_Radius length:sizeof(float) atIndex:13];
    [computeEncoder setBytes:&p_Quality length:sizeof(int) atIndex:14];
    [computeEncoder setBytes:&p_MaskStrength length:sizeof(float) atIndex:15];

    [computeEncoder dispatchThreadgroups:threadGroups threadsPerThreadgroup: threadGroupCount];

    [computeEncoder endEncoding];
    [commandBuffer commit];
    
    // Release textures
    [inputTexture release];
    [maskTexture release];
}