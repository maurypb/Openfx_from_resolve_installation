#!/usr/bin/env python3
"""
OFX Bundle Validator for XML-driven OFX Framework

Validates that the OFX bundle structure meets OpenFX specifications
and checks for common deployment issues.
"""

import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

def validate_bundle_structure(bundle_path="BlurPlugin.ofx.bundle"):
    """Validate OFX bundle directory structure"""
    print(f"Validating bundle structure: {bundle_path}")
    print("=" * 50)
    
    issues = []
    
    # Check bundle directory exists
    if not os.path.exists(bundle_path):
        issues.append(f"Bundle directory '{bundle_path}' does not exist")
        return issues
    
    # Check Contents directory
    contents_dir = os.path.join(bundle_path, "Contents")
    if not os.path.exists(contents_dir):
        issues.append("Missing Contents directory")
    else:
        print("[OK] Contents directory found")
    
    # Check Info.plist
    info_plist = os.path.join(contents_dir, "Info.plist")
    if not os.path.exists(info_plist):
        issues.append("Missing Info.plist file")
    else:
        file_size = os.path.getsize(info_plist)
        if file_size == 0:
            issues.append("Info.plist is empty")
        else:
            print(f"[OK] Info.plist found ({file_size} bytes)")
            
            # Validate Info.plist content
            try:
                tree = ET.parse(info_plist)
                root = tree.getroot()
                if root.tag != 'plist':
                    issues.append("Info.plist is not a valid plist file")
                else:
                    print("[OK] Info.plist is valid XML")
            except ET.ParseError as e:
                issues.append(f"Info.plist XML parse error: {e}")
    
    # Check platform-specific directories
    platforms = ["Win64", "MacOS", "Linux-x86-64"]
    platform_found = False
    
    for platform in platforms:
        platform_dir = os.path.join(contents_dir, platform)
        if os.path.exists(platform_dir):
            platform_found = True
            print(f"[OK] {platform} directory found")
            
            # Check for plugin binary
            if platform == "Win64":
                binary_name = "BlurPlugin.ofx"
            elif platform == "MacOS":
                binary_name = "BlurPlugin.ofx"
            else:  # Linux
                binary_name = "BlurPlugin.ofx"
            
            binary_path = os.path.join(platform_dir, binary_name)
            if os.path.exists(binary_path):
                binary_size = os.path.getsize(binary_path)
                print(f"[OK] Plugin binary found: {binary_name} ({binary_size} bytes)")
            else:
                issues.append(f"Missing plugin binary: {platform}/{binary_name}")
    
    if not platform_found:
        issues.append("No platform-specific directories found (Win64, MacOS, Linux-x86-64)")
    
    return issues

def validate_info_plist_content(bundle_path="BlurPlugin.ofx.bundle"):
    """Validate Info.plist contains required OFX metadata"""
    print("\nValidating Info.plist content...")
    print("=" * 50)
    
    issues = []
    info_plist = os.path.join(bundle_path, "Contents", "Info.plist")
    
    if not os.path.exists(info_plist):
        issues.append("Info.plist not found")
        return issues
    
    try:
        tree = ET.parse(info_plist)
        root = tree.getroot()
        
        # Find the dict element
        dict_elem = root.find('dict')
        if dict_elem is None:
            issues.append("No dict element found in Info.plist")
            return issues
        
        # Extract key-value pairs
        keys = {}
        children = list(dict_elem)
        for i, child in enumerate(children):
            if child.tag == 'key' and i + 1 < len(children):
                key_name = child.text
                next_elem = children[i + 1]
                if next_elem.tag in ['string', 'integer', 'true', 'false']:
                    keys[key_name] = next_elem.text if next_elem.text else next_elem.tag
        
        # Check required keys
        required_keys = [
            'CFBundleExecutable',
            'CFBundleName', 
            'CFBundleIdentifier',
            'CFBundlePackageType'
        ]
        
        for req_key in required_keys:
            if req_key not in keys:
                issues.append(f"Missing required key: {req_key}")
            else:
                print(f"[OK] {req_key}: {keys[req_key]}")
        
        # Check OFX-specific keys
        ofx_keys = [
            'OFXPlugin',
            'OFXPluginVersion',
            'OFXPluginCategory'
        ]
        
        for ofx_key in ofx_keys:
            if ofx_key in keys:
                print(f"[OK] {ofx_key}: {keys[ofx_key]}")
            else:
                print(f"[INFO] Optional OFX key missing: {ofx_key}")
        
        # Validate bundle package type
        if keys.get('CFBundlePackageType') != 'BNDL':
            issues.append(f"CFBundlePackageType should be 'BNDL', got '{keys.get('CFBundlePackageType')}'")
        
    except ET.ParseError as e:
        issues.append(f"Info.plist XML parse error: {e}")
    except Exception as e:
        issues.append(f"Error validating Info.plist: {e}")
    
    return issues

def main():
    print("OFX Bundle Validator")
    print("=" * 50)
    
    # Validate bundle structure
    structure_issues = validate_bundle_structure()
    
    # Validate Info.plist content
    content_issues = validate_info_plist_content()
    
    # Report results
    all_issues = structure_issues + content_issues
    
    print(f"\nValidation Results:")
    print("=" * 50)
    
    if not all_issues:
        print("[SUCCESS] Bundle validation passed!")
        print("The OFX bundle appears to be properly structured.")
        return 0
    else:
        print(f"[ISSUES] Found {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        
        print("\nRecommendations:")
        print("- Ensure the bundle is built completely")
        print("- Check that Info.plist is generated properly")
        print("- Verify platform-specific binaries are copied")
        
        return 1

if __name__ == "__main__":
    sys.exit(main())