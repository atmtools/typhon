#!/bin/bash

rm -rf build
conda build --no-anaconda-upload --output-folder build conda.recipe

