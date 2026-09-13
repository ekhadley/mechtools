#!/bin/sh
cd "$(dirname "$0")"
HF_HUB_OFFLINE=1 uv run pytest "$@"
