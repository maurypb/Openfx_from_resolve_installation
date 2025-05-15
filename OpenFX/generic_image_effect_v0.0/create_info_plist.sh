#!/bin/bash

# Get the directory from the first argument
DIR="$1"

# Create the Info.plist file
cat > "${DIR}/Info.plist" << 'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleDevelopmentRegion</key>
    <string>English</string>
    <key>CFBundleExecutable</key>
    <string>BlurPlugin.ofx</string>
    <key>CFBundleIdentifier</key>
    <string>com.Maury.GaussianBlur</string>
    <key>CFBundleInfoDictionaryVersion</key>
    <string>6.0</string>
    <key>CFBundleName</key>
    <string>GaussianBlur</string>
    <key>CFBundlePackageType</key>
    <string>BNDL</string>
    <key>CFBundleVersion</key>
    <string>1.5</string>
</dict>
</plist>
EOF

echo "Created Info.plist in ${DIR}"