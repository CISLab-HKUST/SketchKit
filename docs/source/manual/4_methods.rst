Methods
=======

This page summarizes the methods supported by SketchKit and links to detailed documentation for each method module.

SketchKit organizes its methods around sketches as the central representation, connecting sketch data with images and 3D shapes. The supported methods can be grouped into three categories:

- **Intra-sketch processing**: Operations where both the input and output are sketches, including cleanup, vectorization, ordering, animation, colorization, and stylization.
- **Sketch-image transformation**: Methods that convert between sketches and images, such as image-to-sketch extraction and sketch-to-image generation.
- **Sketch-shape transformation**: Methods that connect sketches with 3D geometry, including 3D modeling from sketches and line drawing or non-photorealistic rendering from 3D shapes.

.. toctree::
  :maxdepth: 1

  methods/cleanup
  methods/vectorization
  methods/stylization
  methods/ordering
  methods/animation
  methods/image2sketch
  methods/sketch2image
  methods/colorization
  methods/non_photorealistic_renderer
  methods/modeling