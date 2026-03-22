#!/bin/bash

# Ensure at least one argument is provided
if [ "$#" -le 0 ]; then
    echo "Missing command arguments" >&2
    exit 1
fi

if [ "$1" = "streamlit" ]; then
    streamlit "${@:2}"
elif [ "$1" = "pytest" ]; then
    pytest "${@:2}"
else
    python "$1" "${@:2}"
fi