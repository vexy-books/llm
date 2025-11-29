#!/bin/bash
# Build the book with Zensical

set -e

echo "Building The Vexy Book of LLMs..."
zensical build --clean

echo ""
echo "Build complete. Output in ./docs/"
echo "To preview: zensical serve"
