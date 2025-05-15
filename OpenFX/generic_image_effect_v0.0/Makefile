# Updated Makefile for generalized architecture

UNAME_SYSTEM := $(shell uname -s)

CXXFLAGS = -fvisibility=hidden -I../OpenFX-1.4/include -I../Support/include

ifeq ($(UNAME_SYSTEM), Linux)
	AMDAPP_PATH ?= /opt/AMDAPP
	CXXFLAGS += -I${AMDAPP_PATH}/include -fPIC
	CUDAPATH ?= /usr/local/cuda
	NVCC = ${CUDAPATH}/bin/nvcc
	NVCCFLAGS = --compiler-options="-fPIC"
	LDFLAGS = -shared -fvisibility=hidden -L${CUDAPATH}/lib64 -lcuda -lcudart_static
	BUNDLE_DIR = BlurPlugin.ofx.bundle/Contents/Linux-x86-64/
	CUDA_OBJ = CudaKernel.o
	INSTALL_DIR = /usr/OFX/Plugins
else ifeq ($(UNAME_SYSTEM), Darwin)
	ARCH_FLAGS = -arch arm64 -arch x86_64
	CXXFLAGS += ${ARCH_FLAGS}
	LDFLAGS = -bundle -fvisibility=hidden -F/Library/Frameworks -framework OpenCL -framework Metal -framework AppKit
	LDFLAGS += ${ARCH_FLAGS}
	BUNDLE_DIR = BlurPlugin.ofx.bundle/Contents/MacOS/
	METAL_OBJ = MetalKernel.o
	INSTALL_DIR = /Library/OFX/Plugins
else
	# Windows
	CUDAPATH ?= "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0"
	NVCC = "${CUDAPATH}/bin/nvcc.exe"
	NVCCFLAGS = --compiler-options="/MD"
	LDFLAGS = /DLL /SUBSYSTEM:WINDOWS
	BUNDLE_DIR = BlurPlugin.ofx.bundle/Win64/
	CUDA_OBJ = CudaKernel.obj
	INSTALL_DIR = "C:/Program Files/Common Files/OFX/Plugins"
endif

# Framework objects
FRAMEWORK_OBJS = Logger.o GenericImageEffect.o PluginClips.o PluginParameters.o

# Plugin-specific objects
PLUGIN_OBJS = BlurPlugin.o BlurPluginParameters.o BlurPluginMain.o

# OpenFX Support objects
OFX_SUPPORT_OBJS = ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o

# Main target
BlurPlugin.ofx: $(FRAMEWORK_OBJS) $(PLUGIN_OBJS) ${CUDA_OBJ} $(METAL_OBJ) OpenCLKernel.o $(OFX_SUPPORT_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp BlurPlugin.ofx $(BUNDLE_DIR)
	# Create the Contents directory and Info.plist
	mkdir -p $(dir $(BUNDLE_DIR))
	./create_info_plist.sh $(dir $(BUNDLE_DIR))


# Framework compilation
GenericImageEffect.o: GenericImageEffect.cpp GenericImageEffect.h
	$(CXX) -c $< $(CXXFLAGS)

Logger.o: Logger.cpp Logger.h
	$(CXX) -c $< $(CXXFLAGS)

PluginClips.o: PluginClips.cpp PluginClips.h
	$(CXX) -c $< $(CXXFLAGS)

PluginParameters.o: PluginParameters.cpp PluginParameters.h
	$(CXX) -c $< $(CXXFLAGS)

# Plugin-specific compilation
BlurPlugin.o: BlurPlugin.cpp BlurPlugin.h GenericImageEffect.h
	$(CXX) -c $< $(CXXFLAGS)

BlurPluginParameters.o: BlurPluginParameters.cpp BlurPluginParameters.h PluginParameters.h
	$(CXX) -c $< $(CXXFLAGS)

BlurPluginMain.o: BlurPluginMain.cpp BlurPlugin.h BlurPluginParameters.h GenericPluginFactory.h PluginDefines.h
	$(CXX) -c $< $(CXXFLAGS)

# Kernel compilation
CudaKernel.o: CudaKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

MetalKernel.o: MetalKernel.mm
	$(CXX) -c $< $(CXXFLAGS)

OpenCLKernel.o: OpenCLKernel.cpp
	$(CXX) -c $< $(CXXFLAGS)

# OpenFX support library compilation
%.o: ../Support/Library/%.cpp
	$(CXX) -c $< $(CXXFLAGS)

# Clean up build artifacts
clean:
	rm -f *.o *.obj *.ofx
	rm -fr BlurPlugin.ofx.bundle

# Install the plugin to the system-wide OFX plugin directory
install: BlurPlugin.ofx
	mkdir -p $(INSTALL_DIR)
	cp -fr BlurPlugin.ofx.bundle $(INSTALL_DIR)