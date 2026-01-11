"""
Sketch Stylization Module

This module provides a unified API for stylizing sketches using various methods.
"""

from .stylizer import Stylizer

# Also export BrushstrokeStylizer for backward compatibility
from .brushstrokesengine import BrushstrokeStylizer

__all__ = [
    "Stylizer",
    "BrushstrokeStylizer",  # For backward compatibility
]

