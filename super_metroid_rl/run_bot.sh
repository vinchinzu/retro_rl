#!/bin/bash
# Wrapper to run training scripts with uv environment

# Ensure we are in the script directory
cd "$(dirname "$0")"

# Forward arguments to uv run python
uv run python3 "$@"
