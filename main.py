
# External libraries
import numpy as np

# Internal libraries
from objective_function.subclasses.sphere import Sphere
from objective_function.subclasses.rosenbrock import Rosenbrock
from objective_function.subclasses.rastrigin import Rastrigin
from objective_function.subclasses.ackley import Ackley
from objective_function.subclasses.griewank import Griewank
from objective_function.subclasses.michalewicz import Michalewicz
from objective_function.subclasses.styblinski_tang import StyblinskiTang
from objective_function.subclasses.bulkin6 import Bulkin6 
from objective_function.subclasses.dropwave import DropWave

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

    foo = Ackley()
    print(foo.evaluate(np.array([0, 0])))
    foo.apply_shift(np.array([10, 10]))
    print(foo.evaluate(np.array([0, 0])))
    foo.visualize()

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
