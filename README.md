# Purpose

This is a hobby project where I try to implement random ideas in the general domain of graphics programming using low level graphics API and shaders, in this case WebGPU. Currently implemented:

# Hello World Triangle

Basically a template to start other project ideas from.

# Image Viewer

Load images and display them in the viewport, completely implemented with shaders. My focus was on correctly handling sRGB colors, which in practice means that images do not mysteriously get darker when zooming out. In a way, I nerd-sniped myself to this idea because I got irritated that I could not find any image viewer software that does this correctly.

As the whole image processing is implemented in shaders, the viewer is extremely responsive. Additionally, I implemented a dynamic split view when two images get loaded.

