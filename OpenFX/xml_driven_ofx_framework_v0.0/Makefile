# Modified Makefile for the modular OFX plugin with new directory structure

# Directory structure variables
SRC_DIR = src
CORE_DIR = $(SRC_DIR)/core
TESTS_DIR = $(SRC_DIR)/tests
EFFECTS_DIR = effects
TOOLS_DIR = tools
BUILD_DIR = build
OBJ_DIR = $(BUILD_DIR)/obj

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
	CUDA_OBJ = $(EFFECTS_DIR)/TestBlurV2_cuda.o
	OPENCL_OBJ = $(EFFECTS_DIR)/TestBlurV2_opencl.o
	METAL_OBJ = 
	INSTALL_DIR = /usr/OFX/Plugins
else ifeq ($(UNAME_SYSTEM), Darwin)
	ARCH_FLAGS = -arch arm64 -arch x86_64
	CXXFLAGS += ${ARCH_FLAGS}
	LDFLAGS = -bundle -fvisibility=hidden -F/Library/Frameworks -framework OpenCL -framework Metal -framework AppKit
	LDFLAGS += ${ARCH_FLAGS}
	BUNDLE_DIR = BlurPlugin.ofx.bundle/Contents/MacOS/
	CUDA_OBJ = $(EFFECTS_DIR)/TestBlurV2_cuda.o
	OPENCL_OBJ = $(EFFECTS_DIR)/TestBlurV2_opencl.o
	METAL_OBJ = $(EFFECTS_DIR)/TestBlurV2_metal.o
	INSTALL_DIR = /Library/OFX/Plugins
else
	# Windows
	CUDAPATH ?= "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0"
	NVCC = "${CUDAPATH}/bin/nvcc.exe"
	NVCCFLAGS = --compiler-options="/MD"
	LDFLAGS = /DLL /SUBSYSTEM:WINDOWS
	BUNDLE_DIR = BlurPlugin.ofx.bundle/Contents/Win64/
	CUDA_OBJ = $(EFFECTS_DIR)/TestBlurV2_cuda.obj
	OPENCL_OBJ = $(EFFECTS_DIR)/TestBlurV2_opencl.obj
	METAL_OBJ = 
	INSTALL_DIR = "C:/Program Files/Common Files/OFX/Plugins"
endif

# New modular components
PLUGIN_MODULES = Logger.o GenericEffectFactory.o XMLEffectDefinition.o pugixml.o GenericEffect.o GenericProcessor.o ParameterValue.o XMLParameterManager.o XMLInputManager.o KernelWrappers.o KernelRegistry.o

# Main plugin objects
PLUGIN_OBJS = GenericPlugin.o $(PLUGIN_MODULES)

# OpenFX Support objects
OFX_SUPPORT_OBJS = ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o

# Main target
#NOTE - DELETED CL AND METAL FOR NOW: $(METAL_OBJ) $(OPENCL_OBJ)
BlurPlugin.ofx: $(PLUGIN_OBJS) $(CUDA_OBJ)  $(OFX_SUPPORT_OBJS)
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

# Core framework components
GenericEffectFactory.o: $(CORE_DIR)/GenericEffectFactory.cpp $(CORE_DIR)/GenericEffectFactory.h
	$(CXX) -c $< $(CXXFLAGS) -I./$(SRC_DIR) -I./include/pugixml -I./

XMLEffectDefinition.o: $(CORE_DIR)/XMLEffectDefinition.cpp $(CORE_DIR)/XMLEffectDefinition.h
	$(CXX) -c $< $(CXXFLAGS) -I./include/pugixml -I./

XMLParameterManager.o: $(CORE_DIR)/XMLParameterManager.cpp $(CORE_DIR)/XMLParameterManager.h
	$(CXX) -c $< $(CXXFLAGS) -I./$(SRC_DIR) -I./include/pugixml -I./

XMLInputManager.o: $(CORE_DIR)/XMLInputManager.cpp $(CORE_DIR)/XMLInputManager.h
	$(CXX) -c $< $(CXXFLAGS) -I./$(SRC_DIR) -I./include/pugixml -I./

KernelWrappers.o: $(CORE_DIR)/KernelWrappers.cpp $(CORE_DIR)/KernelWrappers.h
	$(CXX) -c $< $(CXXFLAGS) -I./$(SRC_DIR) -I./include/pugixml -I./ -I${CUDAPATH}/include

pugixml.o: include/pugixml/pugixml.cpp
	$(CXX) -c $< $(CXXFLAGS)

GenericEffect.o: $(CORE_DIR)/GenericEffect.cpp $(CORE_DIR)/GenericEffect.h
	$(CXX) -c $< $(CXXFLAGS) -I./$(SRC_DIR) -I./include/pugixml -I./

GenericProcessor.o: $(CORE_DIR)/GenericProcessor.cpp
	$(CXX) -c $< $(CXXFLAGS) -I./$(SRC_DIR) -I./include/pugixml -I./

ParameterValue.o: $(CORE_DIR)/ParameterValue.cpp
	$(CXX) -c $< $(CXXFLAGS) -I./$(SRC_DIR) -I./include/pugixml -I./


# Auto-generate kernel registry from XML files
$(CORE_DIR)/KernelRegistry.cpp $(CORE_DIR)/KernelRegistry.h: $(wildcard $(EFFECTS_DIR)/*.xml)
	python3 $(TOOLS_DIR)/generate_kernel_registry.py

# KernelRegistry compilation (depends on generated files)
KernelRegistry.o: $(CORE_DIR)/KernelRegistry.cpp $(CORE_DIR)/KernelRegistry.h
	$(CXX) -c $< $(CXXFLAGS) -I./$(SRC_DIR) -I./include/pugixml -I./


# Main plugin compilation
GenericPlugin.o: $(CORE_DIR)/GenericPlugin.cpp
	$(CXX) -c $< $(CXXFLAGS) -I./$(SRC_DIR) -I./include/pugixml -I./

# Effect kernel compilation - now in effects directory
$(EFFECTS_DIR)/TestBlurV2_cuda.o: $(EFFECTS_DIR)/TestBlurV2.cu
	${NVCC} -c $< $(NVCCFLAGS) -o $@

# Metal kernel compilation (macOS only)
$(EFFECTS_DIR)/TestBlurV2_metal.o: $(EFFECTS_DIR)/TestBlurV2.metal
	$(CXX) -c $< $(CXXFLAGS) -o $@

# OpenCL kernel compilation  
$(EFFECTS_DIR)/TestBlurV2_opencl.o: $(EFFECTS_DIR)/TestBlurV2.cl
	$(CXX) -c $< $(CXXFLAGS) -o $@

# OpenFX support library compilation
%.o: ../Support/Library/%.cpp
	$(CXX) -c $< $(CXXFLAGS)

# Clean up build artifacts
clean:
	rm -f *.o *.obj *.ofx
	rm -f $(EFFECTS_DIR)/*.o $(EFFECTS_DIR)/*.obj
	rm -fr BlurPlugin.ofx.bundle

# Install the plugin to the system-wide OFX plugin directory
install: BlurPlugin.ofx
	mkdir -p $(INSTALL_DIR)
	cp -fr BlurPlugin.ofx.bundle $(INSTALL_DIR)

# Phony targets
.PHONY: clean install