# High-level implementation goals
- [ ] image viewer
- [ ] Fluid simulation

# Image Viewer
## Ideas
- [x] respect sRGB also on resize / zoom
- [x] comparison of multiple images
- [ ] make an actual usable gui for the viewer

## Immediate Targets
- [x] Render on resize only to texture with viewport size (to fix crash on zoom in)
    - [ ] Reuse previous filter computation on pan
- [x] Replace mipmap with purposefully blurred image based on zoom level

# Simulation
## Ideas
- [ ] Simple 2D rigid body simulation
- [ ] 2D fluid simulation
- [ ] 3D rigid body simulation
- [ ] 3D fluid simulation

## Immediate Targets
- [ ] Add scrollable 2D simulation scene
- [ ] Make object fall onto the floor
