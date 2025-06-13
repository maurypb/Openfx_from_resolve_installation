#!/usr/bin/env python3
"""
Info.plist Generator for XML-driven OFX Framework

Scans the effects/ folder for XML files and generates a proper Info.plist
file for OFX bundle deployment on Windows and other platforms.
"""

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import datetime

def scan_effects_folder(effects_dir="effects"):
    """Scan effects folder for XML files and extract effect information"""
    effects = []
    
    if not os.path.exists(effects_dir):
        print(f"ERROR: Effects directory '{effects_dir}' not found")
        return effects
    
    for xml_file in Path(effects_dir).glob("*.xml"):
        print(f"Processing: {xml_file}")
        
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            if root.tag != 'effect':
                print(f"  WARNING: Skipping {xml_file} - not an effect definition")
                continue
            
            effect_name = root.get('name')
            category = root.get('category', 'Filter')
            
            if not effect_name:
                print(f"  WARNING: Skipping {xml_file} - no effect name")
                continue
            
            # Extract description
            description_elem = root.find('description')
            description = description_elem.text if description_elem is not None else f"{effect_name} OFX Plugin"
            
            effects.append({
                'name': effect_name,
                'category': category,
                'description': description,
                'xml_file': str(xml_file)
            })
            print(f"  Found effect: {effect_name} (Category: {category})")
                
        except ET.ParseError as e:
            print(f"  ERROR: XML parse error in {xml_file}: {e}")
        except Exception as e:
            print(f"  ERROR: Failed to process {xml_file}: {e}")
    
    return effects

def generate_info_plist(effects, bundle_name="BlurPlugin"):
    """Generate Info.plist content for OFX bundle"""
    
    if not effects:
        print("WARNING: No effects found, generating minimal Info.plist")
        primary_effect = {
            'name': 'GenericEffect',
            'category': 'Filter',
            'description': 'Generic OFX Plugin'
        }
    else:
        # Use the first effect as primary
        primary_effect = effects[0]
    
    # Generate current timestamp
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create bundle identifier
    bundle_identifier = f"com.xmlframework.{primary_effect['name']}"
    
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>{bundle_name}</string>
    
    <key>CFBundleName</key>
    <string>{primary_effect['name']}</string>
    
    <key>CFBundleDisplayName</key>
    <string>{primary_effect['name']}</string>
    
    <key>CFBundleIdentifier</key>
    <string>{bundle_identifier}</string>
    
    <key>CFBundleVersion</key>
    <string>1.0.0</string>
    
    <key>CFBundleShortVersionString</key>
    <string>1.0</string>
    
    <key>CFBundlePackageType</key>
    <string>BNDL</string>
    
    <key>CFBundleSignature</key>
    <string>????</string>
    
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    
    <key>CFBundleSupportedPlatforms</key>
    <array>
        <string>Windows</string>
        <string>MacOSX</string>
        <string>Linux</string>
    </array>
    
    <key>NSHumanReadableCopyright</key>
    <string>Copyright © 2025 XML-driven OFX Framework. All rights reserved.</string>
    
    <!-- OFX Specific Properties -->
    <key>OFXPlugin</key>
    <true/>
    
    <key>OFXPluginVersion</key>
    <string>1.4</string>
    
    <key>OFXPluginCategory</key>
    <string>{primary_effect['category']}</string>
    
    <key>OFXPluginDescription</key>
    <string>{primary_effect['description']}</string>
    
    <!-- Effect Information -->
    <key>EffectCount</key>
    <integer>{len(effects)}</integer>
    
    <key>Effects</key>
    <array>"""

    # Add each effect to the plist
    for effect in effects:
        plist_content += f"""
        <dict>
            <key>Name</key>
            <string>{effect['name']}</string>
            <key>Category</key>
            <string>{effect['category']}</string>
            <key>Description</key>
            <string>{effect['description']}</string>
        </dict>"""

    plist_content += f"""
    </array>
    
    <!-- Build Information -->
    <key>BuildTimestamp</key>
    <string>{current_time}</string>
    
    <key>Generator</key>
    <string>XML-driven OFX Framework - generate_info_plist.py</string>
    
</dict>
</plist>
"""
    
    return plist_content

def main():
    print("Info.plist Generator for OFX Bundle")
    print("=" * 50)
    
    # Scan effects folder
    effects = scan_effects_folder()
    
    if effects:
        print(f"\nFound {len(effects)} effect(s):")
        for effect in effects:
            print(f"  - {effect['name']} ({effect['category']})")
    else:
        print("\nNo effects found - generating minimal Info.plist")
    
    # Generate Info.plist content
    plist_content = generate_info_plist(effects)
    
    # Ensure bundle directory exists
    bundle_dir = "BlurPlugin.ofx.bundle/Contents"
    os.makedirs(bundle_dir, exist_ok=True)
    
    # Write Info.plist file
    plist_path = os.path.join(bundle_dir, "Info.plist")
    with open(plist_path, 'w', encoding='utf-8') as f:
        f.write(plist_content)
    
    print(f"\nGenerated: {plist_path}")
    print(f"Bundle ready for deployment!")
    
    # Verify the file was created and has content
    if os.path.exists(plist_path):
        file_size = os.path.getsize(plist_path)
        print(f"File size: {file_size} bytes")
        if file_size > 0:
            print("[OK] Info.plist generated successfully!")
        else:
            print("[ERROR] Info.plist is empty!")
            return 1
    else:
        print("[ERROR] Failed to create Info.plist!")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())