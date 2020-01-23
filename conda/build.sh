#!/bin/bash

rm -rf build
conda-build --output-folder build conda.recipe
conda convert --platform all build/osx-64/typhon-*.bz2 -o build

