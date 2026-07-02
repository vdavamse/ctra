#!/bin/bash
# SageMaker Notebook Instance - On Create Lifecycle Script
# Installs uv and gh CLI tools.
#
# Usage:
#   aws sagemaker create-notebook-instance-lifecycle-config \
#       --notebook-instance-lifecycle-config-name ctra-tools \
#       --on-create Content=$(base64 -w0 scripts/sagemaker/on-create.sh)

set -euo pipefail

echo "=== CTRA: Installing developer tools ==="

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install gh
GH_VERSION=$(curl -s https://api.github.com/repos/cli/cli/releases/latest | grep -oP '"tag_name": "v\K[^"]+')
curl -sL "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_amd64.tar.gz" | tar xz -C /tmp
cp "/tmp/gh_${GH_VERSION}_linux_amd64/bin/gh" /usr/local/bin/

# Add uv to PATH
echo 'export PATH="/home/ec2-user/.local/bin:${PATH}"' >> /home/ec2-user/.bashrc

echo "=== CTRA: Tools installed (uv, gh) ==="
