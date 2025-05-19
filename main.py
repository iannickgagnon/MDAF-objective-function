# External libraries
import numpy as np

from MDAF_objective_functions.implementations.ackley import Ackley
from MDAF_objective_functions.implementations.bukin6 import Bukin6
from MDAF_objective_functions.implementations.dropwave import DropWave
from MDAF_objective_functions.implementations.eggholder import Eggholder
from MDAF_objective_functions.implementations.franke import Franke
from MDAF_objective_functions.implementations.gramacy_and_lee import GramacyAndLee
from MDAF_objective_functions.implementations.griewank import Griewank
from MDAF_objective_functions.implementations.holder_table import HolderTable
from MDAF_objective_functions.implementations.langermann import Langermann
from MDAF_objective_functions.implementations.levy import Levy
from MDAF_objective_functions.implementations.michalewicz import Michalewicz
from MDAF_objective_functions.implementations.rastrigin import Rastrigin
from MDAF_objective_functions.implementations.rosenbrock import Rosenbrock
from MDAF_objective_functions.implementations.six_hump_camel_back import (
    SixHumpCamelBack,
)

# Internal libraries
from MDAF_objective_functions.implementations.sphere import Sphere
from MDAF_objective_functions.implementations.styblinski_tang import StyblinskiTang

if __name__ == "__main__":

    # Instantiate objective function
    StyblinskiTang().visualize()

    """
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

    """
    from scipy.optimize import minimize
    
    foo = Langermann()
    bounds = [(0, 10), (0, 10)]
    result = minimize(foo.evaluate, x0=[1, 10], bounds=bounds)

    # Print the minimum value and the corresponding x and y
    print(f"Minimum value: {result.fun}")
    print(f"At position: {result.x}")
    """
