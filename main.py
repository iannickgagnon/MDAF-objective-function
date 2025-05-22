import numpy as np

from MDAF_benchmarks import Ackley
from MDAF_benchmarks import Bukin6
from MDAF_benchmarks import DropWave
from MDAF_benchmarks import Eggholder
from MDAF_benchmarks import Franke
from MDAF_benchmarks import GramacyAndLee
from MDAF_benchmarks import Griewank
from MDAF_benchmarks import HolderTable
from MDAF_benchmarks import Langermann
from MDAF_benchmarks import Levy
from MDAF_benchmarks import Michalewicz
from MDAF_benchmarks import Rastrigin
from MDAF_benchmarks import Rosenbrock
from MDAF_benchmarks import SixHumpCamelBack
from MDAF_benchmarks import Sphere
from MDAF_benchmarks import StyblinskiTang

import MDAF_benchmarks as of

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
