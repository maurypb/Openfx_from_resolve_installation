# One-Click Deployment for TestBlurV2 OFX Plugin

## 🚀 Quick Start

The VSCode tasks have been configured for complete one-click build and deployment to DaVinci Resolve.

### Available Tasks

#### 1. **Build and Deploy to DaVinci Resolve** ⭐ *Recommended*
- **Shortcut**: `Ctrl+Shift+P` → "Tasks: Run Task" → "Build and Deploy to DaVinci Resolve"
- **What it does**: Complete pipeline from source to DaVinci Resolve
  1. Creates build directories
  2. Generates kernel signature from XML
  3. Generates kernel registry
  4. Compiles C++ sources
  5. Compiles CUDA kernel
  6. Links OFX plugin
  7. Generates Info.plist
  8. Copies to bundle structure
  9. Deploys to global OFX directory
  10. Shows success message with next steps

#### 2. **Deploy to DaVinci Resolve**
- **Use when**: Plugin is already built, just need to deploy
- **What it does**: Copies existing bundle to DaVinci Resolve

#### 3. **Verify Installation**
- **Use when**: Want to check if plugin is properly installed
- **What it does**: Verifies files exist and shows other installed OFX plugins

#### 4. **Build OFX Plugin (Windows MSVC)**
- **Use when**: Just want to build without deploying
- **What it does**: Builds plugin but doesn't deploy

## 🎯 One-Click Workflow

### Step 1: Build and Deploy
```
Ctrl+Shift+P → Tasks: Run Task → "Build and Deploy to DaVinci Resolve"
```

### Step 2: Restart DaVinci Resolve
- Close DaVinci Resolve completely
- Restart the application

### Step 3: Test the Plugin
1. Open/create a project
2. Go to Color or Edit page
3. Find **TestBlurV2** in Effects Library under **Filter** category
4. Drag onto a video clip
5. Test parameters:
   - **Blur Radius** (0-100)
   - **Quality** (1-32)
   - **Mask Strength** (0-1)
   - **Brightness** (0-2) - Test parameter
   - **Redness** (-1 to 2) - Test parameter

## 🔧 Task Dependencies

The build system automatically handles dependencies:

```
Build and Deploy to DaVinci Resolve
└── Deploy to DaVinci Resolve
    └── Post-Build Steps
        ├── Build OFX Plugin (Windows MSVC)
        │   └── Link OFX Plugin
        │       ├── Compile C++ Sources
        │       │   ├── Create Build Directories
        │       │   └── Generate Kernel Signature
        │       │       └── Generate Kernel Registry
        │       └── Compile CUDA Kernel
        └── Generate Info.plist
```

## 📁 What Gets Deployed

The deployment copies the complete bundle structure:

```
C:\Program Files\Common Files\OFX\Plugins\BlurPlugin.ofx.bundle\
├── Contents\
│   ├── Info.plist (658 bytes)
│   └── Win64\
│       └── BlurPlugin.ofx (777,728 bytes)
```

## 🚨 Troubleshooting

### Task Fails with Permission Error
- Run VSCode as Administrator
- Or manually copy using the provided batch script

### Plugin Doesn't Appear in DaVinci Resolve
1. Run "Verify Installation" task to check files
2. Ensure DaVinci Resolve was completely restarted
3. Check DaVinci Resolve's OFX plugin logs

### Build Errors
- Ensure CUDA Toolkit 12.9 is installed
- Verify Visual Studio 2022 Community is installed
- Check that all paths in tasks.json are correct

## 📋 Quick Verification

After deployment, run the "Verify Installation" task to confirm:
- [x] Plugin binary exists (777,728 bytes)
- [x] Info.plist exists (658 bytes)
- [x] Bundle structure is correct
- [x] Other OFX plugins are visible

## 🎉 Success Indicators

You'll know the deployment worked when:
1. Build task completes without errors
2. Deployment shows "DEPLOYMENT COMPLETE" message
3. Verification task shows all checkmarks
4. TestBlurV2 appears in DaVinci Resolve Effects Library
5. Effect applies to video clips without errors

---

**Framework**: XML-Driven OFX Framework v0.0  
**Platform**: Windows 64-bit  
**Target**: DaVinci Resolve (OFX 1.4)