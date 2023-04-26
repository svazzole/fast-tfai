#! /bin/bash

if hash podman 2>/dev/null; then
        podman run -i \
                    --security-opt=label=disable \
                    --hooks-dir=/usr/share/containers/oci/hooks.d/ \
                    -v /path/to/dataset/:/data \
                    -v /path/to/yaml:/app/conf/params.yaml \
                    localhost/inspair:v0.0.3
    else
        docker run -i \
                    -v /path/to/dataset/:/data \
                    -v /path/to/yaml:/app/conf/params.yaml \
                    localhost/inspair:v0.0.3

fi
