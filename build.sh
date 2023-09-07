#!/bin/bash
export LD_LIBRARY_PATH=./so:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=./so/out:$LD_LIBRARY_PATH
sh ./src/builder.sh
# make
