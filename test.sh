#!/bin/sh
# Static checks (undefined names, redefinitions, unused locals, comparisons to None/True), then the tests, offline against the local HF cache.
cd "$(dirname "$0")"
uv run ruff check --select F821,F811,F841,E711,E712 src tests && HF_HUB_OFFLINE=1 uv run pytest "$@"
