Methods
=======

This page summarizes the methods supported by SketchKit and links to detailed documentation for each method module.

SketchKit organizes its methods around sketches as the central representation, connecting sketch data with images and 3D shapes. The supported methods can be grouped into three categories:

- **Intra-sketch processing**: Operations where both the input and output are sketches, including cleanup, vectorization, stylization, ordering, and animation.
- **Sketch-image transformation**: Methods that convert between sketches and images, such as image-to-sketch extraction, sketch-to-image generation, and colorization.
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
  methods/non_photorealistic_rendering
  methods/modeling


How to contribute a new method to SketchKit
---------------------------
You can integrate a new method into SketchKit following the :doc:`step-by-step document <5_method_contribution>`.