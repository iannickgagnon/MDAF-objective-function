
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
from src.objective_functions.subclasses.bukin6 import Bukin6 
from src.objective_functions.subclasses.dropwave import DropWave
from src.objective_functions.subclasses.eggholder import Eggholder
from src.objective_functions.subclasses.gramacy_and_lee import GramacyAndLee
from src.objective_functions.subclasses.holder_table import HolderTable
from src.objective_functions.subclasses.franke import Franke
from src.objective_functions.subclasses.six_hump_camel_back import SixHumpCamelBack
from src.objective_functions.subclasses.langermann import Langermann
from src.objective_functions.subclasses.levy import Levy

if __name__ == "__main__":

    foo =  Rosenbrock()

    foo.visualize()
    

    '''
    foo = DropWave()
    foo.profile(nb_calls=100, 
                nb_positions=int(1e6), 
                filename='profile.txt')
    '''
    
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
    
    foo = Langermann()
    bounds = [(0, 10), (0, 10)]
    result = minimize(foo.evaluate, x0=[1, 10], bounds=bounds)

    # Print the minimum value and the corresponding x and y
    print(f"Minimum value: {result.fun}")
    print(f"At position: {result.x}")
    '''
