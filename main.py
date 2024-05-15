
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

if __name__ == "__main__":

    '''
    Ackley().visualize()
    Griewank().visualize()
    Michalewicz().visualize()
    StyblinskiTang().visualize()
    Bulkin6().visualize()
    DropWave().visualize()
    Sphere().visualize()
    Rosenbrock().visualize()  
    Rastrigin().visualize()  
    '''

    foo = GramacyAndLee()
    foo.apply_noise(0, 0.5)
    foo.apply_shift(np.array([-2]))
    foo.visualize()
    print(foo.nb_calls)

    '''
    foo.visualize() 
    print(foo.parallel_evaluate(np.array([[1, 1], [2, 2]])))
    foo.apply_shift(np.array([5, 5]))
    foo.apply_noise(0, 0.1)
    foo.visualize()
    print(foo.parallel_evaluate(np.array([[1, 1], [2, 2]])))
    '''
