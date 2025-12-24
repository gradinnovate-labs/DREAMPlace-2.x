#!/bin/bash

# Parse arguments
if [ "$1" = "--remove" ]; then
    echo "Removing existing dreamplace2:cuda image..."

    # Remove any containers using the image first
    CONTAINERS=$(docker ps -a -q -f ancestor=dreamplace2:cuda 2>/dev/null)
    if [ -n "$CONTAINERS" ]; then
        echo "Removing containers using dreamplace2:cuda..."
        docker rm $CONTAINERS
    fi

    # Now remove the image
    docker rmi dreamplace2:cuda 2>/dev/null && echo "Image removed successfully." || echo "No existing image to remove."
else
    docker build . --file Dockerfile --tag dreamplace2:cuda
fi
