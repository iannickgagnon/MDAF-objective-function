import numpy as np

from MDAF_objective_functions import Ackley
from MDAF_objective_functions import Bukin6
from MDAF_objective_functions import DropWave
from MDAF_objective_functions import Eggholder
from MDAF_objective_functions import Franke
from MDAF_objective_functions import GramacyAndLee
from MDAF_objective_functions import Griewank
from MDAF_objective_functions import HolderTable
from MDAF_objective_functions import Langermann
from MDAF_objective_functions import Levy
from MDAF_objective_functions import Michalewicz
from MDAF_objective_functions import Rastrigin
from MDAF_objective_functions import Rosenbrock
from MDAF_objective_functions import SixHumpCamelBack
from MDAF_objective_functions import Sphere
from MDAF_objective_functions import StyblinskiTang

import MDAF_objective_functions as of

if __name__ == "__main__":

    """
    # Advanced functionality: Profiling
    foo = DropWave()
    foo.profile(nb_calls=100, 
                nb_positions=int(1e6), 
                filename='profile.txt')
    """

    # foo.apply_noise(0, 0.5)
    # foo.apply_shift(np.array([0, 0]))
    # foo.visualize()
    # print(foo.nb_calls)
    # print(foo.evaluate(np.array([[1, 1]])))
    # print(foo.evaluate(np.array([[0.47, 0.8]])))

    """
    foo.visualize() 
    print(foo.parallel_evaluate(np.array([[1, 1], [2, 2]])))
    foo.apply_shift(np.array([5, 5]))
    foo.apply_noise(0, 0.1)
    foo.visualize()
    print(foo.parallel_evaluate(np.array([[1, 1], [2, 2]])))
    """
