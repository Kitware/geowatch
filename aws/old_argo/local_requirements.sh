#Install argo CLI
mkdir -p "$HOME/tmp/argo"
cd "$HOME/tmp/argo"

# Download the binary
curl -sLO https://github.com/argoproj/argo-workflows/releases/download/v3.2.6/argo-linux-amd64.gz

# Unzip
gunzip argo-linux-amd64.gz

# Make binary executable
chmod +x argo-linux-amd64

# Move binary to path
PREFIX=$HOME/.local
cp ./argo-linux-amd64 "$PREFIX/bin/argo"

# Test installation
argo version
