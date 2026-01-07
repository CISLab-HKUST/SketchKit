#!/bin/bash
# Setup script to copy brushstroke_engine code to third_party

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
SKETCHKIT_ROOT="$( cd "$SCRIPT_DIR/../.." && pwd )"
TARGET_DIR="$SCRIPT_DIR"

# Default brushstroke_engine path
DEFAULT_BRUSHSTROKE_PATH="$HOME/developer/brushstroke_engine/brushstroke_engine"

echo "=========================================="
echo "Brushstroke Engine Setup"
echo "=========================================="
echo ""

# Check if brushstroke_engine path is provided
if [ -z "$1" ]; then
    BRUSHSTROKE_PATH="$DEFAULT_BRUSHSTROKE_PATH"
    echo "No path provided, using default: $BRUSHSTROKE_PATH"
else
    BRUSHSTROKE_PATH="$1"
fi

# Check if source directory exists
if [ ! -d "$BRUSHSTROKE_PATH" ]; then
    echo "❌ Error: brushstroke_engine directory not found at: $BRUSHSTROKE_PATH"
    echo ""
    echo "Please provide the path to brushstroke_engine:"
    echo "  $0 /path/to/brushstroke_engine/brushstroke_engine"
    echo ""
    echo "Or clone it first:"
    echo "  git clone https://github.com/NVlabs/brushstroke_engine.git"
    echo "  $0 /path/to/brushstroke_engine/brushstroke_engine"
    exit 1
fi

# Check if forger directory exists
if [ ! -d "$BRUSHSTROKE_PATH/forger" ]; then
    echo "❌ Error: forger directory not found in $BRUSHSTROKE_PATH"
    exit 1
fi

# Check if thirdparty directory exists
if [ ! -d "$BRUSHSTROKE_PATH/thirdparty" ]; then
    echo "❌ Error: thirdparty directory not found in $BRUSHSTROKE_PATH"
    exit 1
fi

echo "✅ Found brushstroke_engine at: $BRUSHSTROKE_PATH"
echo "📁 Target directory: $TARGET_DIR"
echo ""

# Create target directory
mkdir -p "$TARGET_DIR"

# Copy forger directory
echo "📦 Copying forger/ directory..."
if [ -d "$TARGET_DIR/forger" ]; then
    echo "   Warning: forger/ already exists, backing up..."
    mv "$TARGET_DIR/forger" "$TARGET_DIR/forger.backup.$(date +%Y%m%d_%H%M%S)"
fi
cp -r "$BRUSHSTROKE_PATH/forger" "$TARGET_DIR/"
echo "   ✅ Copied forger/"

# Copy thirdparty directory
echo "📦 Copying thirdparty/ directory..."
if [ -d "$TARGET_DIR/thirdparty" ]; then
    echo "   Warning: thirdparty/ already exists, backing up..."
    mv "$TARGET_DIR/thirdparty" "$TARGET_DIR/thirdparty.backup.$(date +%Y%m%d_%H%M%S)"
fi
cp -r "$BRUSHSTROKE_PATH/thirdparty" "$TARGET_DIR/"
echo "   ✅ Copied thirdparty/"

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Install sketchkit: cd $SKETCHKIT_ROOT && pip install -e ."
echo "  2. Verify installation: python -c 'import forger; import thirdparty.stylegan2_ada_pytorch.dnnlib; print(\"✅ Success!\")'"
echo ""

