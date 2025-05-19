# Enhanced XML Schema for OFX Framework

This updated schema incorporates valuable features from the Autodesk shader format while maintaining our clean, attribute-based approach for the OFX framework.

## Basic Structure with Enhanced Metadata

```xml
<effect name="EffectName" category="Category" 
       version="1.0" 
       description="Detailed description of the effect"
       author="Author Name"
       copyright="Copyright information"
       supports_timeline="true"
       supports_matte="true">
  
  <!-- Define input sources with border modes -->
  <inputs>
    <source name="source" label="Main Image" border_mode="clamp" />
    <source name="matte" label="Matte Input" optional="true" border_mode="black" />
  </inputs>
  
  <!-- Parameters with enhanced controls -->
  <parameters>
    <!-- Basic numeric parameter with increment control -->
    <parameter name="radius" type="double" default="5.0" min="0.0" max="100.0" 
               displayMin="0.0" displayMax="50.0" inc="0.1"
               res_dependent="none"
               label="Radius" hint="Blur radius in pixels" />
    
    <!-- Integer parameter -->
    <parameter name="quality" type="int" default="8" min="1" max="32" 
               displayMin="1" displayMax="16" inc="1"
               label="Quality" hint="Number of samples for the blur" />
    
    <!-- Boolean parameter -->
    <parameter name="invert" type="bool" default="false"
               label="Invert" hint="Invert the effect" />
    
    <!-- Color parameter with individual component controls -->
    <parameter name="main_color" type="color" label="Main Color" 
               hint="The primary color of the effect"
               value_type="colour">
      <component name="r" default="0.3" min="0.0" max="1.0" inc="0.01" />
      <component name="g" default="0.5" min="0.0" max="1.0" inc="0.01" />
      <component name="b" default="0.7" min="0.0" max="1.0" inc="0.01" />
    </parameter>
    
    <!-- Vector2 parameter -->
    <parameter name="offset" type="vec2" label="Offset" 
               hint="XY position offset">
      <component name="x" default="0.0" min="-100.0" max="100.0" inc="0.5" />
      <component name="y" default="0.0" min="-100.0" max="100.0" inc="0.5" />
    </parameter>
    
    <!-- Enhanced curve parameter -->
    <parameter name="alpha_fade" type="curve" default_shape="ease_out"
               curve_background="0" 
               label="Alpha Fade" 
               hint="Controls alpha falloff from inside to outside" />
    
    <!-- Choice parameter with options -->
    <parameter name="filter_type" type="choice" default="0"
               label="Filter Type" hint="Type of filter to apply">
      <option value="0" label="Box" />
      <option value="1" label="Triangle" />
      <option value="2" label="Gaussian" />
    </parameter>
    
    <!-- String parameter -->
    <parameter name="note" type="string" default=""
               label="Note" hint="User notes for this effect" />
  </parameters>
  
  <!-- Enhanced UI organization -->
  <ui>
    <page name="Main" tooltip="Primary controls">
      <column name="Basic" tooltip="Basic parameters">
        <parameter>radius</parameter>
        <parameter>quality</parameter>
        <parameter>invert</parameter>
      </column>
      <column name="Colors" tooltip="Color controls">
        <parameter>main_color</parameter>
        <parameter>alpha_fade</parameter>
      </column>
    </page>
    <page name="Advanced" tooltip="Advanced controls">
      <column name="Position" tooltip="Position controls">
        <parameter>offset</parameter>
      </column>
      <column name="Filters" tooltip="Filter settings">
        <parameter>filter_type</parameter>
      </column>
    </page>
  </ui>
  
  <!-- Identity conditions define when the effect is a pass-through -->
  <identity_conditions>
    <condition>
      <parameter name="radius" operator="lessEqual" value="0.0" />
    </condition>
  </identity_conditions>
  
  <!-- Version 1: Single kernel processing -->
  <kernels>
    <cuda file="EffectKernel.cu" executions="1" />
    <opencl file="EffectKernel.cl" executions="1" />
    <metal file="EffectKernel.metal" executions="1" />
  </kernels>
  
  <!-- Version 2: Multi-kernel pipeline (replaces the kernels section) -->
  <!-- <pipeline>
    <step name="EdgeDetect" executions="1">
      <kernels>
        <cuda file="EdgeDetect.cu" />
        <opencl file="EdgeDetect.cl" />
        <metal file="EdgeDetect.metal" />
      </kernels>
    </step>
    
    <step name="Blur" executions="3">
      <kernels>
        <cuda file="GaussianBlur.cu" />
        <opencl file="GaussianBlur.cl" />
        <metal file="GaussianBlur.metal" />
      </kernels>
    </step>
    
    <step name="Composite" executions="1">
      <kernels>
        <cuda file="Composite.cu" />
        <opencl file="Composite.cl" />
        <metal file="Composite.metal" />
      </kernels>
    </step>
  </pipeline> -->
</effect>
```

## Key Enhancements from Autodesk Format

1. **Enhanced Metadata**
   - Added `version`, `author`, `copyright` attributes to the effect tag
   - Added `supports_timeline` and `supports_matte` capability flags

2. **Parameter Improvements**
   - Added `inc` attribute to control increment step in UI
   - Added `res_dependent` attribute to indicate if values scale with resolution
   - Enhanced tooltips with more detailed descriptions

3. **Vector Component Controls**
   - Replaced simple min/max for vectors with component-specific controls
   - Added individual `<component>` tags for each vector component
   - Each component can have its own min, max, default, and increment values

4. **Enhanced Curve Parameters**
   - Added `default_shape` to specify initial curve shape
   - Added `curve_background` for UI display

5. **Color Parameter Type**
   - Added dedicated color type with RGB components
   - Each component has individual min/max/inc controls

6. **UI Enhancements**
   - Added tooltips to pages and columns
   - Better organization for complex parameter sets

7. **Choice Parameters**
   - Enhanced with clearer label/value separation

These improvements provide a more user-friendly and flexible parameter system while maintaining the clean, attribute-based approach of our original schema.

## Example: Gaussian Blur with Enhanced Parameters

```xml
<effect name="GaussianBlur" category="Filter" 
       version="1.0" 
       description="Apply Gaussian blur with optional mask control"
       author="Maury Rosenfeld"
       copyright="Copyright 2025">
  
  <inputs>
    <source name="source" label="Input Image" border_mode="clamp" />
    <source name="matte" label="Mask" optional="true" border_mode="black" />
  </inputs>
  
  <parameters>
    <parameter name="radius" type="double" default="5.0" min="0.0" max="100.0" 
               displayMin="0.0" displayMax="50.0" inc="0.1" 
               res_dependent="width"
               label="Radius" hint="Blur radius in pixels" />
               
    <parameter name="quality" type="int" default="8" min="1" max="32" 
               displayMin="1" displayMax="16" inc="1" 
               label="Quality" hint="Number of samples for the blur" />
               
    <parameter name="edge_mode" type="choice" default="0"
               label="Edge Behavior" hint="How to handle pixels at the edges">
      <option value="0" label="Repeat Edge Pixels" />
      <option value="1" label="Black Beyond Edges" />
      <option value="2" label="Tile (Repeat Pattern)" />
    </parameter>
    
    <parameter name="blur_rgb" type="bool" default="true"
               label="Blur RGB" hint="Apply blur to color channels" />
               
    <parameter name="blur_alpha" type="bool" default="true"
               label="Blur Alpha" hint="Apply blur to alpha channel" />
  </parameters>
  
  <ui>
    <page name="Main">
      <column name="Blur">
        <parameter>radius</parameter>
        <parameter>quality</parameter>
      </column>
      <column name="Options">
        <parameter>edge_mode</parameter>
        <parameter>blur_rgb</parameter>
        <parameter>blur_alpha</parameter>
      </column>
    </page>
  </ui>
  
  <identity_conditions>
    <condition>
      <parameter name="radius" operator="lessEqual" value="0.0" />
    </condition>
  </identity_conditions>
  
  <kernels>
    <cuda file="GaussianBlur.cu" executions="1" />
    <opencl file="GaussianBlur.cl" executions="1" />
    <metal file="GaussianBlur.metal" executions="1" />
  </kernels>
</effect>
```

This enhanced schema maintains compatibility with our previously defined architecture while adding the valuable UI and parameter control features from the Autodesk shader format.
