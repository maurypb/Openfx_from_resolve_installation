#include "GenericImageEffect.h"
#include "ofxsImageEffect.h"
#include "ofxsMultiThread.h"
#include <cstring>

GenericImageProcessor::GenericImageProcessor(OFX::ImageEffect& p_Instance)
    : _effect(p_Instance)
    , _dstImg(nullptr)
    , _srcImg(nullptr)
    , _maskImg(nullptr)
    , _pOpenCLCmdQ(nullptr)
    , _pCudaStream(nullptr)
    , _pMetalCmdQ(nullptr)
{
}

void GenericImageProcessor::setGPURenderArgs(const OFX::RenderArguments& p_Args)
{
    // Access GPU resources through proper struct members
    // These might vary based on OFX version - we're using the standard names
    _pOpenCLCmdQ = p_Args.pOpenCLCmdQ;
    
#ifndef __APPLE__
    _pCudaStream = p_Args.pCudaStream;
#endif
    
#ifdef __APPLE__
    _pMetalCmdQ = p_Args.pMetalCmdQ;
#endif
}

void GenericImageProcessor::process()
{
    // Check if we have GPU support - this is version and host dependent
    // For now, we'll use a more basic approach that works with older OFX versions
    bool useGPU = false;
    
    // Try OpenCL
    if (_pOpenCLCmdQ && supportsOpenCL()) {
        processImagesOpenCL();
        return;
    }
    
    // Try CUDA on non-Apple systems
#ifndef __APPLE__
    if (_pCudaStream && supportsCUDA()) {
        processImagesCUDA();
        return;
    }
#endif
    
    // Try Metal on Apple systems
#ifdef __APPLE__
    if (_pMetalCmdQ && supportsMetal()) {
        processImagesMetal();
        return;
    }
#endif
    
    // Fallback to CPU processing
    processCPU();
}

bool GenericImageProcessor::supportsOpenCL() const
{
    // Default implementation - derived classes can override
    return false;
}

bool GenericImageProcessor::supportsCUDA() const
{
    // Default implementation - derived classes can override
    return false;
}

bool GenericImageProcessor::supportsMetal() const
{
    // Default implementation - derived classes can override
    return false;
}

void GenericImageProcessor::processCPU()
{
    // Create a processor for multi-threaded rendering
    class ProcessorThreading : public OFX::MultiThread::Processor
    {
    public:
        ProcessorThreading(GenericImageProcessor& processor, OfxRectI window)
            : _processor(processor), _window(window) {}
        
        // Implementation of pure virtual function from OFX::MultiThread::Processor
        virtual void multiThreadFunction(unsigned int threadID, unsigned int nThreads) override
        {
            // Split the render window into slices, one per thread
            OfxRectI window = _window;
            unsigned int height = window.y2 - window.y1;
            unsigned int linesPerThread = height / nThreads;
            unsigned int start = window.y1 + threadID * linesPerThread;
            unsigned int end = threadID == nThreads - 1 ? window.y2 : start + linesPerThread;
            
            if (end > start) {
                OfxRectI threadWindow = window;
                threadWindow.y1 = start;
                threadWindow.y2 = end;
                
                // Process this slice
                _processor.multiThreadProcessImages(threadWindow);
            }
        }
        
    private:
        GenericImageProcessor& _processor;
        OfxRectI _window;
    };
    
    // Create and run the processor
    ProcessorThreading processor(*this, _renderWindow);
    processor.multiThread();
}