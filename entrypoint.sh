#!/bin/bash

set -e

if [ "$1" = 'serve' ]; then
    echo "Starting server..."
    python3 /app/server.py
else
    echo "Not implemented"
    exit 1
fi
