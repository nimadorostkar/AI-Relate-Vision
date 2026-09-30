#!/usr/bin/env bash
# ./run.sh [videos or folders] [options]   - see ./run.sh --help
DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ ! -x "$DIR/.venv/bin/python" ]]; then
    echo "Not set up yet - run ./setup_mac.sh first." >&2
    exit 1
fi
exec "$DIR/.venv/bin/python" "$DIR/relate_video.py" "$@"
