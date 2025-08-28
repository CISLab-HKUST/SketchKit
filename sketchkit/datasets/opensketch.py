import os
import wget

import xml.etree.ElementTree as ET
from svg.path import parse_path, path

from sketchkit.core.sketch import Sketch, Path, Curve, Vertex, Point
from sketchkit.utils.general_utils.md5sum import *
from sketchkit.utils.general_utils.md5sum_table import *
from sketchkit.utils.general_utils.extract_files import *
from sketchkit.core import Sketch

