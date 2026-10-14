#!/usr/bin/env bash

set -euo pipefail

# Keep this default equal to ARG AUTOILLUSTRATE_VERSION in
# src/docker/agent-cli/Dockerfile; tests/test_investigation_cli.py checks the two agree.
AUTOILLUSTRATE_VERSION="${AUTOILLUSTRATE_VERSION:-0.4.8}"
INSTALL_PREFIX="${AUTOILLUSTRATE_PREFIX:-${HOME}/.local}"
INSTALL_DRAWIO=0

usage() {
  cat <<'EOF'
Usage: scripts/install_autoillustrate.sh [--install-drawio-with-nix]

Installs the pinned autoillustrate npm release into a user-local npm prefix.
Set AUTOILLUSTRATE_PREFIX to override the default (~/.local) and
AUTOILLUSTRATE_VERSION to install a different release.

Options:
  --install-drawio-with-nix  Install pinned Draw.io packages with an existing Nix.
  -h, --help                  Show this help.
EOF
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --install-drawio-with-nix) INSTALL_DRAWIO=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

case "$INSTALL_PREFIX" in
  /) echo "Refusing to install into filesystem root" >&2; exit 1 ;;
  /*) ;;
  *) echo "AUTOILLUSTRATE_PREFIX must be an absolute path" >&2; exit 1 ;;
esac
if [[ ! "$AUTOILLUSTRATE_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "AUTOILLUSTRATE_VERSION must be an exact release such as 0.4.8" >&2
  exit 1
fi

command -v node >/dev/null 2>&1 || { echo "Node.js >= 22 is required" >&2; exit 1; }
command -v npm >/dev/null 2>&1 || { echo "npm is required" >&2; exit 1; }
if [ "$INSTALL_DRAWIO" -eq 1 ]; then
  if [ "$(uname -s)" != "Linux" ]; then
    echo "--install-drawio-with-nix is Linux-only" >&2
    echo "On macOS: brew install --cask drawio" >&2
    exit 1
  fi
  command -v nix >/dev/null 2>&1 || {
    echo "--install-drawio-with-nix requires an existing Nix installation" >&2
    exit 1
  }
fi

node_major="$(node -p 'process.versions.node.split(".")[0]')"
if [[ ! "$node_major" =~ ^[0-9]+$ ]] || [ "$node_major" -lt 22 ]; then
  echo "Node.js >= 22 is required (found $(node --version 2>/dev/null || echo unknown))" >&2
  exit 1
fi

npm install --global --prefix "$INSTALL_PREFIX" "autoillustrate@${AUTOILLUSTRATE_VERSION}"

AUTOILLUSTRATE_BIN="$INSTALL_PREFIX/bin/autoillustrate"
if [ ! -x "$AUTOILLUSTRATE_BIN" ]; then
  echo "Autoillustrate installation did not create $AUTOILLUSTRATE_BIN" >&2
  exit 1
fi
installed_version="$("$AUTOILLUSTRATE_BIN" --version)"
if [ "$installed_version" != "$AUTOILLUSTRATE_VERSION" ]; then
  echo "Installed autoillustrate reports $installed_version, expected $AUTOILLUSTRATE_VERSION" >&2
  exit 1
fi
echo "$installed_version"

AUTOILLUSTRATE_PACKAGE_DIR="$INSTALL_PREFIX/lib/node_modules/autoillustrate"
if [ "$INSTALL_DRAWIO" -eq 1 ]; then
  bash "$AUTOILLUSTRATE_PACKAGE_DIR/scripts/install_drawio.sh"
elif ! command -v drawio >/dev/null 2>&1; then
  echo "Autoillustrate installed; Draw.io is still required for native standalone use." >&2
  if [ "$(uname -s)" = "Darwin" ]; then
    echo "Install it with: brew install --cask drawio" >&2
  else
    echo "Re-run with --install-drawio-with-nix, or install Draw.io separately." >&2
  fi
fi

case ":$PATH:" in
  *":$INSTALL_PREFIX/bin:"*) ;;
  *) echo "Add $INSTALL_PREFIX/bin to PATH to use autoillustrate." ;;
esac
