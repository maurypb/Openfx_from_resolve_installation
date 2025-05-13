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

# Main target
BlurPlugin.ofx: BlurPlugin.o ${CUDA_OBJ} $(METAL_OBJ) OpenCLKernel.o ofxsCore.o ofxsImageEffect.o ofxsInteract.o ofxsLog.o ofxsMultiThread.o ofxsParams.o ofxsProperty.o ofxsPropertyValidation.o
	$(CXX) $^ -o $@ $(LDFLAGS)
	mkdir -p $(BUNDLE_DIR)
	cp BlurPlugin.ofx $(BUNDLE_DIR)

# CUDA kernel compilation
CudaKernel.o: CudaKernel.cu
	${NVCC} -c $< $(NVCCFLAGS)

# Metal kernel compilation (macOS only)
MetalKernel.o: MetalKernel.mm
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