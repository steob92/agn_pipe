#!/bin/bash

[[ "$1" ]] && IMAGE_NAME=$1  || IMAGE_NAME="agn_pipe.sif"


if command -v apptainer &> /dev/null
then
    apptainer build ./$IMAGE_NAME agn_pipe.def
else
    singularity build ./$IMAGE_NAME agn_pipe.def
fi
