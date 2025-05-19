```xml
<effect name="EffectName" category="Category">
  <description>Effect description text</description>
  
  <!-- Define input sources with border mode -->
  <inputs>
    <source name="source" label="Main Image" border_mode="clamp" />
    <source name="matte" label="Matte Input" optional="true" border_mode="black" />
    <!-- Additional source inputs can be defined here -->
  </inputs>
  
  <!-- Parameters define UI controls and processing values -->
  <parameters>
    <parameter name="radius" type="double" default="5.0" min="0.0" max="100.0" 
               displayMin="0.0" displayMax="50.0" label="Radius" hint="Blur radius in pixels" />
    
    <parameter name="quality" type="int" default="8" min="1" max="32" 
               displayMin="1" displayMax="16" label="Quality" hint="Number of samples for the blur" />
    
    <parameter name="alpha_fade" type="curve" default_shape="ease_out"
               label="Alpha Fade" hint="Controls alpha falloff from inside to outside" />
    
    <!-- Additional parameters... -->
  </parameters>
  
  <!-- UI organization -->
  <ui>
    <page name="Main">
      <column name="Basic">
        <parameter>radius</parameter>
        <parameter>quality</parameter>
      </column>
      <column name="Advanced">
        <parameter>alpha_fade</parameter>
      </column>
    </page>
    <page name="Color">
      <!-- More parameters -->
    </page>
  </ui>
  
  <!-- Identity conditions define when the effect is a pass-through -->
  <identity_conditions>
    <condition>
      <parameter name="radius" operator="lessEqual" value="0.0" />
    </condition>
    <!-- Additional conditions... -->
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