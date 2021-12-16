#!/bin/bash

rm -rf build
conda-build --output-folder build conda.recipe

