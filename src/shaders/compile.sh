#!/bin/bash

mkdir -p target/shaders
glslang -V100 src/shaders/triangle.vert -o target/shaders/vert.spv
glslang -V100 src/shaders/triangle.frag -o target/shaders/frag.spv
