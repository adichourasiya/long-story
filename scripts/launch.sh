#!/usr/bin/env bash
# launch.sh
#
# Activates the project's .venv if not already active, installs dependencies
# using `uv` when available (falls back to pip/pyproject/requirements), and
# launches the interactive program `novel_ai.py`.
#
# Usage (bash/zsh):
#   ./scripts/launch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Color output functions
info() { echo -e "\033[0;36m$1\033[0m"; }
success() { echo -e "\033[0;32m$1\033[0m"; }
warning() { echo -e "\033[0;33m$1\033[0m"; }
error() { echo -e "\033[0;31m$1\033[0m"; exit 1; }

# 1) Activate .venv if not already
VENV_ACTIVATE="$PROJECT_ROOT/.venv/bin/activate"
if [ -z "$VIRTUAL_ENV" ]; then
    if [ -f "$VENV_ACTIVATE" ]; then
        info "Activating virtual environment from $VENV_ACTIVATE"
        source "$VENV_ACTIVATE" || warning "Activation script returned non-zero; continuing."
    else
        warning ".venv not found at $VENV_ACTIVATE - continuing without activation."
    fi
else
    info "Virtual environment already active: $VIRTUAL_ENV"
fi

# Helper function to run python
run_python() {
    # Use python3 on macOS/Linux, fall back to python if python3 doesn't exist
    if command -v python3 &>/dev/null; then
        python3 "$@"
    else
        python "$@"
    fi
    return $?
}

# 2) Ensure pip exists (try python -m pip, ensurepip, or get-pip.py)
info 'Ensuring pip is available...'
if ! run_python -m pip --version &>/dev/null; then
    info 'Bootstrapping pip via ensurepip (if available)...'
    if ! run_python -m ensurepip --upgrade &>/dev/null; then
        info 'ensurepip failed or not available; attempting get-pip.py download...'
        TMP_FILE=$(mktemp /tmp/get-pip.XXXXXX.py)
        if curl -sSL https://bootstrap.pypa.io/get-pip.py -o "$TMP_FILE"; then
            run_python "$TMP_FILE" || warning "get-pip.py install failed"
            rm -f "$TMP_FILE"
        else
            warning "Failed to download get-pip.py"
        fi
    fi
fi

# 3) Install dependencies: prefer `uv` if present, otherwise fall back to pip
info 'Installing project dependencies...'
if command -v uv &>/dev/null; then
    info "Found 'uv' CLI at $(command -v uv). Running 'uv sync'"
    if uv sync; then
        success 'Dependencies installed via uv.'
    else
        warning "'uv sync' failed with exit code $? - falling back to pip."
        UV_FAILED=1
    fi
else
    UV_FAILED=1
fi

# Helper function to test if a package is installed
test_package_installed() {
    if command -v python3 &>/dev/null; then
        python3 -c "import $1" &>/dev/null
    else
        python -c "import $1" &>/dev/null
    fi
    return $?
}

if [ -n "$UV_FAILED" ]; then
    # Prefer pyproject.toml editable install when available
    PYPROJECT="$PROJECT_ROOT/pyproject.toml"
    REQS="$PROJECT_ROOT/requirements.txt"
    
    if [ -f "$PYPROJECT" ]; then
        info "Installing package in-place from pyproject.toml (editable)."
        cd "$PROJECT_ROOT"
        if ! run_python -m pip install -e .; then
            warning "pip editable install failed"
        fi
    elif [ -f "$REQS" ]; then
        info "Installing from requirements.txt"
        if ! run_python -m pip install -r "$REQS"; then
            warning "pip install -r requirements.txt failed"
        fi
    else
        info 'No pyproject.toml or requirements.txt found - skipping dependency install.'
    fi
fi

# After attempting installation, ensure prompt_toolkit is available
info 'Verifying prompt_toolkit is installed...'
if ! test_package_installed 'prompt_toolkit'; then
    if command -v uv &>/dev/null; then
        info "Attempting to install 'prompt_toolkit' with 'uv run'..."
        uv run python -m pip install prompt_toolkit || warning "'uv run' attempt failed"
    fi
    
    if ! test_package_installed 'prompt_toolkit'; then
        info "Falling back to pip to install prompt_toolkit..."
        run_python -m pip install prompt_toolkit
    fi
    
    if ! test_package_installed 'prompt_toolkit'; then
        error "prompt_toolkit is required for the interactive CLI but could not be installed. Please install it manually: pip install prompt_toolkit"
    else
        success "prompt_toolkit is installed."
    fi
else
    success "prompt_toolkit is already installed."
fi

# 4) Launch the program
info 'Launching interactive CLI: python novel_ai.py'
cd "$PROJECT_ROOT"
if command -v python3 &>/dev/null; then
    exec python3 novel_ai.py
else
    exec python novel_ai.py
fi
