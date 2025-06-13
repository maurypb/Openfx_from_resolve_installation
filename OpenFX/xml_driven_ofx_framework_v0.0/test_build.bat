@echo off
echo ==========================================
echo Testing Fixed VSCode Build Tasks
echo ==========================================
echo.

echo Step 1: Creating build directories...
if not exist "build\obj" mkdir "build\obj"
if not exist "BlurPlugin.ofx.bundle\Contents\Win64" mkdir "BlurPlugin.ofx.bundle\Contents\Win64"
echo [✓] Build directories created

echo.
echo Step 2: Generating kernel registry...
python tools/generate_kernel_registry.py
if %ERRORLEVEL% NEQ 0 (
    echo [✗] Kernel registry generation failed
    exit /b 1
)
echo [✓] Kernel registry generated

echo.
echo Step 3: Compiling C++ sources...
"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/cl.exe" /c /EHsc /std:c++17 /MD /favor:AMD64 /DWIN64 /D_WIN64 /Iz:/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/OpenFX-1.4/include /Iz:/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/Support/include /I. /I./include /I./include/pugixml /I./src/core /I"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/include" /Fo:build/obj/ Logger.cpp src/core/GenericEffect.cpp src/core/GenericEffectFactory.cpp src/core/GenericPlugin.cpp src/core/GenericProcessor.cpp src/core/KernelRegistry.cpp src/core/KernelWrappers.cpp src/core/ParameterValue.cpp src/core/XMLEffectDefinition.cpp src/core/XMLInputManager.cpp src/core/XMLParameterManager.cpp include/pugixml/pugixml.cpp z:/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/Support/Library/ofxsCore.cpp z:/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/Support/Library/ofxsImageEffect.cpp z:/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/Support/Library/ofxsInteract.cpp z:/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/Support/Library/ofxsLog.cpp z:/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/Support/Library/ofxsMultiThread.cpp z:/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/Support/Library/ofxsParams.cpp z:/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/Support/Library/ofxsProperty.cpp z:/PROJECTS/SOFTWARE_PROJECTS/ofx/Starting_again_250504/Openfx_from_resolve_installation/OpenFX/Support/Library/ofxsPropertyValidation.cpp > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [✗] C++ compilation failed
    exit /b 1
)
echo [✓] C++ sources compiled

echo.
echo Step 4: Compiling CUDA kernel...
"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/bin/nvcc.exe" -c .\effects\TestBlurV2.cu -o .\build\obj\TestBlurV2_cuda.obj --compiler-bindir "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64" -Xcompiler /MD,/favor:AMD64,/DWIN64,/D_WIN64 -m64 > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [✗] CUDA compilation failed
    exit /b 1
)
echo [✓] CUDA kernel compiled

echo.
echo Step 5: Linking OFX plugin (CRITICAL TEST)...
"C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/bin/Hostx64/x64/link.exe" /DLL /MACHINE:X64 /SUBSYSTEM:WINDOWS /OUT:build/BlurPlugin.ofx build/obj/Logger.obj build/obj/GenericEffect.obj build/obj/GenericEffectFactory.obj build/obj/GenericPlugin.obj build/obj/GenericProcessor.obj build/obj/KernelRegistry.obj build/obj/KernelWrappers.obj build/obj/ParameterValue.obj build/obj/XMLEffectDefinition.obj build/obj/XMLInputManager.obj build/obj/XMLParameterManager.obj build/obj/pugixml.obj build/obj/ofxsCore.obj build/obj/ofxsImageEffect.obj build/obj/ofxsInteract.obj build/obj/ofxsLog.obj build/obj/ofxsMultiThread.obj build/obj/ofxsParams.obj build/obj/ofxsProperty.obj build/obj/ofxsPropertyValidation.obj build/obj/TestBlurV2_cuda.obj "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/lib/x64/cudart.lib" "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/lib/x64/msvcrt.lib" "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/lib/x64/msvcprt.lib" "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207/lib/x64/vcruntime.lib" "C:/Program Files (x86)/Windows Kits/10/lib/10.0.26100.0/um/x64/kernel32.lib" "C:/Program Files (x86)/Windows Kits/10/lib/10.0.26100.0/um/x64/user32.lib" "C:/Program Files (x86)/Windows Kits/10/lib/10.0.26100.0/um/x64/gdi32.lib" "C:/Program Files (x86)/Windows Kits/10/lib/10.0.26100.0/um/x64/advapi32.lib" "C:/Program Files (x86)/Windows Kits/10/lib/10.0.26100.0/um/x64/shell32.lib" "C:/Program Files (x86)/Windows Kits/10/lib/10.0.26100.0/um/x64/ole32.lib" "C:/Program Files (x86)/Windows Kits/10/lib/10.0.26100.0/um/x64/oleaut32.lib" "C:/Program Files (x86)/Windows Kits/10/lib/10.0.26100.0/um/x64/uuid.lib" "C:/Program Files (x86)/Windows Kits/10/lib/10.0.26100.0/ucrt/x64/ucrt.lib" > nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [✗] LINKING FAILED - This was the original problem!
    exit /b 1
)
echo [✓] LINKING SUCCESSFUL - Problem fixed!

echo.
echo Step 6: Verifying plugin creation...
if exist build\BlurPlugin.ofx (
    echo [✓] BlurPlugin.ofx created successfully
    dir build\BlurPlugin.ofx
) else (
    echo [✗] BlurPlugin.ofx not found
    exit /b 1
)

echo.
echo ==========================================
echo [✓] ALL TESTS PASSED!
echo ==========================================
echo The linking command fix is working perfectly!
echo VSCode tasks should now work without the cmd.exe wrapper issues.