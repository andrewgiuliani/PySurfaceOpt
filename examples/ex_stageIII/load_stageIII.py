#!/usr/bin/env python3
import numpy as np
import pysurfaceopt as pys
import sys
import os

# This example simply loads the coils after the five and nine stage optimization runs discussed in
#
# A. Giuliani, F. Wechsung, M. Landreman, G. Stadler, A. Cerfon, Direct computation of magnetic surfaces in Boozer coordinates and coil optimization for quasi-symmetry. Submitted.
#
# instatiates an optimization object and prints the objective and its gradient.

for coilset in ['five', 'nine']:
    for length in [18, 20, 22, 24]:
        problem = pys.get_stageIII_problem(coilset=coilset, length=length)
        print(f"{coilset} surface optimization, length={length}: J(x) = {problem.res:3e} dJ(x) = {np.linalg.norm(problem.dres, ord=np.inf):.3e}")
    print("---------------------------------------------------------------------------------------------------------------------------------")
