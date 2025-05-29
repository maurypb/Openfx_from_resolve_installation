# Modified Makefile for the modular OFX plugin

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

# New modular components
PLUGIN_MODULES = Logger.o PluginClips.o PluginParameters.o BlurPluginParameters.o GenericEffectFactory.o XMLEffectDefinition.o pugixml.o GenericEffect.o GenericProcessor.o ParameterValue.o XMLParameterManager.o XMLInputManager.o KernelWrappers.o KernelWrappers.o

# Main plugin objects
PLUGIN_OBJS = BlurPlugin.o $(PLUGIN_MODULES)

# OpenFX Support objects
OFX_SUPPORT_OBJS = ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o

# Main target
BlurPlugin.ofx: $(PLUGIN_OBJS) ${CUDA_OBJ} $(METAL_OBJ) OpenCLKernel.o $(OFX_SUPPORT_OBJS)
	$(CXX) $^ -o $@ $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp BlurPlugin.ofx $(BUNDLE_DIR)
	mkdir -p BlurPlugin.ofx.bundle/Contents
	echo '<?xml version="1.0" encoding="UTF-8"?>' > BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '<plist version="1.0">' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '<dict>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <key>CFBundleDevelopmentRegion</key>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <string>English</string>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <key>CFBundleExecutable</key>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <string>BlurPlugin.ofx</string>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <key>CFBundleIdentifier</key>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <string>com.xmlframework.TestBlurV2</string>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <key>CFBundleInfoDictionaryVersion</key>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <string>6.0</string>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <key>CFBundleName</key>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <string>TestBlurV2</string>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <key>CFBundlePackageType</key>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <string>BNDL</string>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <key>CFBundleVersion</key>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '    <string>1.5</string>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '</dict>' >> BlurPlugin.ofx.bundle/Contents/Info.plist
	echo '</plist>' >> BlurPlugin.ofx.bundle/Contents/Info.plist

# New modular components compilation
Logger.o: Logger.cpp Logger.h
	$(CXX) -c $< $(CXXFLAGS)

PluginClips.o: PluginClips.cpp PluginClips.h
	$(CXX) -c $< $(CXXFLAGS)

PluginParameters.o: PluginParameters.cpp PluginParameters.h
	$(CXX) -c $< $(CXXFLAGS)

BlurPluginParameters.o: BlurPluginParameters.cpp BlurPluginParameters.h
	$(CXX) -c $< $(CXXFLAGS)

# NEW COMPILATION RULES - THESE WERE MISSING
GenericEffectFactory.o: src/core/GenericEffectFactory.cpp src/core/GenericEffectFactory.h
	$(CXX) -c $< $(CXXFLAGS) -I./src -I./include/pugixml

XMLEffectDefinition.o: src/core/XMLEffectDefinition.cpp src/core/XMLEffectDefinition.h
	$(CXX) -c $< $(CXXFLAGS) -I./include/pugixml


# MISSING COMPILATION RULES ADDED
XMLParameterManager.o: src/core/XMLParameterManager.cpp src/core/XMLParameterManager.h
	$(CXX) -c $< $(CXXFLAGS) -I./src -I./include/pugixml

XMLInputManager.o: src/core/XMLInputManager.cpp src/core/XMLInputManager.h
	$(CXX) -c $< $(CXXFLAGS) -I./src -I./include/pugixml

KernelWrappers.o: src/core/KernelWrappers.cpp src/core/KernelWrappers.h
	$(CXX) -c $< $(CXXFLAGS) -I./src -I./include/pugixml

KernelWrappers.o: src/core/KernelWrappers.cpp src/core/KernelWrappers.h
	$(CXX) -c $< $(CXXFLAGS) -I./src -I./include/pugixml -I${CUDAPATH}/include

pugixml.o: include/pugixml/pugixml.cpp
	$(CXX) -c $< $(CXXFLAGS)

GenericEffect.o: src/core/GenericEffect.cpp src/core/GenericEffect.h
	$(CXX) -c $< $(CXXFLAGS) -I./src -I./include/pugixml

GenericProcessor.o: src/core/GenericProcessor.cpp
	$(CXX) -c $< $(CXXFLAGS) -I./src -I./include/pugixml

ParameterValue.o: src/core/ParameterValue.cpp
	$(CXX) -c $< $(CXXFLAGS) -I./src -I./include/pugixml



# Main plugin compilation
BlurPlugin.o: BlurPlugin.cpp BlurPlugin.h
	$(CXX) -c $< $(CXXFLAGS)

# CUDA kernel compilation
CudaKernel.o: CudaKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

# Metal kernel compilation (macOS only)
MetalKernel.o: MetalKernel.mm
	$(CXX) -c $< $(CXXFLAGS)

# OpenCL kernel compilation
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

# Phony targets
.PHONY: clean install