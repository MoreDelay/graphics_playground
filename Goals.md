# High-level implementation goals
- [ ] image viewer
- [ ] Fluid simulation

# Image Viewer
## Ideas
- [x] respect sRGB also on resize / zoom
- [ ] comparison of multiple images
- [ ] make an actual usable gui for the viewer

## Immediate targets
- [x] Render on resize only to texture with viewport size (to fix crash on zoom in)
    - [ ] Reuse previous filter computation on pan
- [x] Replace mipmap with purposefully blurred image based on zoom level
