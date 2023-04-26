#! /bin/bash

if hash podman 2>/dev/null; then
        podman build -f Dockerfile -t inspair:v0.0.3
else
        docker build -f Dockerfile -t inspair:v0.0.3
fi
