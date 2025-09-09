#!/bin/bash

# Initialize meshai-core repository
# This script sets up the initial commit for the meshai-core repository

set -e

echo "========================================="
echo "Initializing MeshAI Core Repository"
echo "========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if we're in the right directory
if [ ! -f "setup.py" ] || [ ! -d "meshai_core" ]; then
    echo "Error: This script must be run from the meshai-core directory"
    exit 1
fi

# Initialize git if not already done
if [ ! -d ".git" ]; then
    git init
    echo -e "${GREEN}✓${NC} Git repository initialized"
fi

# Add all files
git add .
echo -e "${GREEN}✓${NC} Files staged"

# Create initial commit
git commit -m "Initial commit: MeshAI Core v1.0.0

- Add shared API key validator library
- Set up Python package structure
- Add setup.py for pip installation
- Include comprehensive validation features:
  - 16-character prefix validation
  - Bcrypt hash verification
  - Multi-layer caching (local + Redis)
  - Circuit breaker pattern
  - Support for user, service, and admin keys"

echo -e "${GREEN}✓${NC} Initial commit created"

# Tag the version
git tag -a v1.0.0 -m "Version 1.0.0: Initial release with API key validator"
echo -e "${GREEN}✓${NC} Tagged as v1.0.0"

# Add remote (you'll need to update this with your actual GitHub URL)
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Create a repository on GitHub named 'meshai-core'"
echo "2. Add the remote:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/meshai-core.git"
echo "3. Push to GitHub:"
echo "   git push -u origin main"
echo "   git push origin v1.0.0"
echo ""
echo "Repository is ready for push!"