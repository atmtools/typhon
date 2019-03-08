"""Functions related to colors."""
from .common import *

import matplotlib.colors as mcolors


TYPHON_COLORS = (
    ('jetblack', '#343434'),
    ('chalmers-blue', '#024a86'),
    ('chalmers-grey', '#b9cbcf'),
    ('darkgrey', '#767676'),
    ('eigengrau', '#16161d'),
    ('grey', '#b0b0b0'),
    ('max-planck', '#006c66'),
    ('uhh-red', '#ee1d23'),
    ('uhh-blue', '#00a1d7'),
    ('uhh-grey', '#314e58'),
)

# Add "ty:" prefix to prevent name collision with original matplotlib colors
TYPHON_COLORS = {'ty:' + name: value for name, value in TYPHON_COLORS}
mcolors.get_named_colors_mapping().update(TYPHON_COLORS)
