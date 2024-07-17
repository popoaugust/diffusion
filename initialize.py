from . import generation_recombination
from ..utils import parameters, grid, matrices

def initialize():
    params = parameters.Params()
    grid = grid.Grid(params)
    mat = matrices.Matrices(params, grid)
    genrec = generation_recombination.GR(params, grid)
    
    return params, grid, mat, genrec

params = parameters.Params()
grid = grid.Grid(params)
mat = matrices.Matrices(params, grid)
genrec = generation_recombination.GR(params, grid)

initialize()

