# TestBlurV2 OFX Plugin - Deployment and Testing Guide

## ✅ Installation Status

**SUCCESSFULLY DEPLOYED** - The TestBlurV2 OFX plugin has been installed to the global OFX plugins directory.

### Installation Details
- **Plugin Location**: `C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle`
- **Binary Size**: 777,728 bytes (777 KB)
- **Installation Date**: June 11, 2025
- **Bundle Structure**: Complete with Info.plist and Win64 binary

## 🏗️ Plugin Architecture

### XML-Driven Framework
This plugin demonstrates the XML-driven OFX framework with:
- **Effect Definition**: [`effects/TestBlurV2.xml`](effects/TestBlurV2.xml)
- **CUDA Kernel**: [`effects/TestBlurV2.cu`](effects/TestBlurV2.cu)
- **OpenCL Kernel**: [`effects/TestBlurV2.cl`](effects/TestBlurV2.cl)
- **Metal Kernel**: [`effects/TestBlurV2.metal`](effects/TestBlurV2.metal)

### Plugin Parameters
The TestBlurV2 effect provides the following controls:

#### Basic Controls Group
- **Blur Radius**: 0.0 - 100.0 pixels (default: 30.0)
- **Quality**: 1 - 32 samples (default: 8)

#### Masking Group
- **Mask Strength**: 0.0 - 1.0 (default: 1.0)

#### Test Group
- **Brightness**: 0.0 - 2.0 (default: 1.0) - *Test parameter*
- **Redness**: -1.0 - 2.0 (default: 1.0) - *Test parameter*

## 🧪 Testing Instructions for DaVinci Resolve

### Step 1: Launch DaVinci Resolve
1. **Close DaVinci Resolve** if it's currently running
2. **Restart DaVinci Resolve** to ensure it scans for new OFX plugins
3. Wait for the application to fully load

### Step 2: Locate the TestBlurV2 Effect
1. Open or create a project in DaVinci Resolve
2. Go to the **Color** page or **Edit** page
3. In the **Effects Library** panel, look for:
   - **Category**: Filter
   - **Effect Name**: TestBlurV2
   - **Description**: "Test blur effect for GenericEffectFactory validation"

### Step 3: Apply the Effect
1. **Drag and drop** TestBlurV2 onto a video clip
2. The effect should appear in the **Node Graph** (Color page) or **Inspector** (Edit page)
3. **Expected Result**: The effect applies without errors

### Step 4: Test Parameters
Verify that all XML-defined parameters are available:

#### Basic Controls
- [ ] **Blur Radius** slider (0-100, default 30)
- [ ] **Quality** slider (1-32, default 8)

#### Masking
- [ ] **Mask Strength** slider (0-1, default 1.0)

#### Test Group
- [ ] **Brightness** slider (0-2, default 1.0)
- [ ] **Redness** slider (-1 to 2, default 1.0)

### Step 5: Functional Testing
1. **Adjust Blur Radius**: Should blur the image progressively
2. **Change Quality**: Should affect blur smoothness
3. **Test Identity Condition**: Set Blur Radius to 0 - should pass through unchanged
4. **Apply Mask**: Connect a mask input if available

## 🔍 Verification Steps

### Plugin Loading Verification
1. **Check DaVinci Resolve Console** (if accessible):
   - Look for OFX plugin loading messages
   - Verify no error messages related to BlurPlugin

2. **Effect Availability**:
   - TestBlurV2 appears in Effects Library
   - Effect can be applied to clips
   - Parameters are responsive

### Bundle Structure Verification
```
C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle\
├── Contents\
│   ├── Info.plist (658 bytes)
│   └── Win64\
│       └── BlurPlugin.ofx (777,728 bytes)
```

### File Verification Commands
```cmd
dir "C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle\Contents\Win64\BlurPlugin.ofx"
dir "C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle\Contents\Info.plist"
```

## 🚨 Troubleshooting

### Plugin Not Appearing in DaVinci Resolve

1. **Restart DaVinci Resolve**: Ensure complete application restart
2. **Check File Permissions**: Verify the plugin files are readable
3. **Verify Bundle Structure**: Ensure all files are in correct locations
4. **Check DaVinci Resolve Version**: Ensure OFX compatibility

### Plugin Loads but Doesn't Function

1. **Check CUDA Availability**: Ensure CUDA runtime is available
2. **Verify GPU Compatibility**: Check if GPU supports required features
3. **Review DaVinci Resolve Logs**: Look for runtime errors

### Manual Reinstallation

If needed, reinstall the plugin:

```cmd
# Remove existing installation
rmdir /s "C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle"

# Reinstall from project directory
robocopy BlurPlugin.ofx.bundle "C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle" /E
```

## 📋 Installation Checklist

- [x] Plugin binary built successfully (777,728 bytes)
- [x] Bundle structure created with Info.plist
- [x] Plugin copied to global OFX directory
- [x] File permissions verified
- [ ] DaVinci Resolve restarted
- [ ] Effect appears in Effects Library
- [ ] Parameters function correctly
- [ ] Blur effect processes video correctly

## 🎯 Success Criteria

The deployment is successful when:

1. **TestBlurV2** appears in DaVinci Resolve's Effects Library under **Filter** category
2. Effect can be **applied to video clips** without errors
3. All **XML-defined parameters** are available and functional
4. **Blur processing** works correctly with CUDA acceleration
5. **Identity condition** (radius = 0) passes video unchanged

## 📝 Next Steps

After successful testing:

1. **Document any issues** encountered during testing
2. **Verify XML parameter mapping** works as expected
3. **Test with different video formats** and resolutions
4. **Validate CUDA kernel performance** on target hardware
5. **Prepare for additional effect definitions** using the same framework

---

**Framework Version**: XML-Driven OFX Framework v0.0  
**Build Date**: June 11, 2025  
**Platform**: Windows 64-bit  
**OFX Version**: 1.4  
**CUDA Version**: 12.9