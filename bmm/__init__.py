
"""bmm: Bayesian Map-matching"""

from bmm.src.tools.plot import plot

from bmm.src.inference.smc import initiate_particles
from bmm.src.inference.smc import update_particles
from bmm.src.inference.smc import offline_map_match
from bmm.src.inference.smc import _offline_map_match_fl

from bmm.src.inference.smc import updates
from bmm.src.inference.smc import proposals

from bmm.src.inference.model import MapMatchingModel
from bmm.src.inference.model import GammaMapMatchingModel

from bmm.src.inference.proposal import get_possible_routes

from bmm.src.tools.edges import cartesianise_path
from bmm.src.tools.edges import get_geometry
from bmm.src.tools.edges import discretise_edge

from bmm.src.inference.particles import MMParticles

# try:
#   del src
# except NameError:
#   pass
