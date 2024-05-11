
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

    foo = Sphere()

    foo.visualize()
    #foo.apply_shift(np.array([1, 1]))
    #foo.apply_noise(np.random.randn)
    #print(foo.parallel_evaluate(np.array([[1, 1], [2, 2]])))
    #foo.visualize()
    print(foo.parallel_evaluate(np.array([[1, 1], [2, 2]])))

    '''
    import timeit
    import numpy as np

    # Measure the execution time of evaluate_a
    time_a = timeit.timeit(lambda: ackley.evaluate(np.random.uniform(-5, 5, 2)), number=10000)

    # Measure the execution time of evaluate_b
    time_b = timeit.timeit(lambda: ackley.evaluate_b(np.random.uniform(-5, 5, 2)), number=10000)

    print(f"Execution time of evaluate: {time_a} seconds")
    print(f"Execution time of evaluate_b: {time_b} seconds")
    '''
