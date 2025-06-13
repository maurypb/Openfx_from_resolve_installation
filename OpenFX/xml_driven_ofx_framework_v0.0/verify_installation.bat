@echo off
echo ========================================
echo TestBlurV2 OFX Plugin Installation Verification
echo ========================================
echo.

echo Checking global OFX plugins directory...
if exist "C:\Program Files\Common Files\OFX\Plugins" (
    echo [✓] Global OFX directory exists
) else (
    echo [✗] Global OFX directory not found
    goto :error
)

echo.
echo Checking BlurPlugin bundle...
if exist "C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle" (
    echo [✓] BlurPlugin.ofx.bundle exists
) else (
    echo [✗] BlurPlugin.ofx.bundle not found
    goto :error
)

echo.
echo Checking bundle structure...
if exist "C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle\Contents\Info.plist" (
    echo [✓] Info.plist exists
) else (
    echo [✗] Info.plist missing
    goto :error
)

if exist "C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle\Contents\Win64\BlurPlugin.ofx" (
    echo [✓] BlurPlugin.ofx binary exists
) else (
    echo [✗] BlurPlugin.ofx binary missing
    goto :error
)

echo.
echo Checking file sizes...
for %%F in ("C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle\Contents\Win64\BlurPlugin.ofx") do (
    if %%~zF==777728 (
        echo [✓] BlurPlugin.ofx size correct: %%~zF bytes
    ) else (
        echo [!] BlurPlugin.ofx size unexpected: %%~zF bytes (expected: 777728)
    )
)

for %%F in ("C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle\Contents\Info.plist") do (
    if %%~zF==658 (
        echo [✓] Info.plist size correct: %%~zF bytes
    ) else (
        echo [!] Info.plist size unexpected: %%~zF bytes (expected: 658)
    )
)

echo.
echo Listing other OFX plugins for reference...
dir "C:\Program Files\Common Files\OFX\Plugins" /b

echo.
echo ========================================
echo [✓] INSTALLATION VERIFICATION COMPLETE
echo ========================================
echo.
echo Next steps:
echo 1. Restart DaVinci Resolve completely
echo 2. Look for "TestBlurV2" in Effects Library under Filter category
echo 3. Apply effect to a video clip and test parameters
echo.
echo If the effect doesn't appear, check DaVinci Resolve's OFX plugin logs.
echo.
pause
goto :end

:error
echo.
echo ========================================
echo [✗] INSTALLATION VERIFICATION FAILED
echo ========================================
echo.
echo Please check the installation and try again.
echo Refer to DEPLOYMENT_GUIDE.md for troubleshooting steps.
echo.
pause

:end