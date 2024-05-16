
# External libraries
import numpy as np

# Internal libraries
from src.objective_functions.subclasses.sphere import Sphere
from src.objective_functions.subclasses.rosenbrock import Rosenbrock
from src.objective_functions.subclasses.rastrigin import Rastrigin
from src.objective_functions.subclasses.ackley import Ackley
from src.objective_functions.subclasses.griewank import Griewank
from src.objective_functions.subclasses.michalewicz import Michalewicz
from src.objective_functions.subclasses.styblinski_tang import StyblinskiTang
from src.objective_functions.subclasses.bulkin6 import Bulkin6 
from src.objective_functions.subclasses.dropwave import DropWave
from src.objective_functions.subclasses.eggholder import Eggholder
from src.objective_functions.subclasses.gramacy_and_lee import GramacyAndLee
from src.objective_functions.subclasses.holder_table import HolderTable
from src.objective_functions.subclasses.franke import Franke
from src.objective_functions.subclasses.six_hump_camel_back import SixHumpCamelBack
from src.objective_functions.subclasses.langermann import Langermann

if __name__ == "__main__":

    foo = Langermann().visualize()

    #foo.apply_noise(0, 0.5)
    #foo.apply_shift(np.array([0, 0]))
    #foo.visualize()
    #print(foo.nb_calls)

    #print(foo.evaluate(np.array([[1, 1]])))
    #print(foo.evaluate(np.array([[0.47, 0.8]])))

    '''
    foo.visualize() 
    print(foo.parallel_evaluate(np.array([[1, 1], [2, 2]])))
    foo.apply_shift(np.array([5, 5]))
    foo.apply_noise(0, 0.1)
    foo.visualize()
    print(foo.parallel_evaluate(np.array([[1, 1], [2, 2]])))
    '''

    '''
    from scipy.optimize import minimize
    bounds = [(0, 1), (0, 1)]
    result = minimize(foo.evaluate, x0=[0.5, 0.5], bounds=bounds)

    # Print the minimum value and the corresponding x and y
    print(f"Minimum value: {result.fun}")
    print(f"At position: {result.x}")
    '''
