
# External libraries
import numpy as np

# Internal libraries
from objective_function.subclasses.sphere import Sphere
from objective_function.subclasses.rosenbrock import Rosenbrock
from objective_function.subclasses.rastrigin import Rastrigin
from objective_function.subclasses.ackley import Ackley
from objective_function.subclasses.griewank import Griewank
from objective_function.subclasses.michalewicz import Michalewicz

'''
if __name__ == "__main__":

    # Create an instance of the Sphere objective function
    sphere = Sphere(dimensionality=2, search_space_bounds=[(-5, 5), (-5, 5)])

    # Visualize the objective function in 2D
    sphere.visualize(dimensions=[0, 1], bounds=[(-5, 5), (-5, 5)], resolution=100)
'''

'''
if __name__ == "__main__":

    # Create an instance of the Rosenbrock objective function
    rosenbrock = Rosenbrock(dimensionality=2, search_space_bounds=[(-5, 5), (-5, 5)])

    # Visualize the objective function in 2D
    rosenbrock.visualize(dimensions=[0, 1], bounds=[(-5, 5), (-5, 5)], resolution=100)
'''

'''
if __name__ == "__main__":

    # Create an instance of the Rastrigin objective function
    rastrigin = Rastrigin(dimensionality=2, search_space_bounds=[(-5, 5), (-5, 5)])

    # Visualize the objective function in 2D
    rastrigin.visualize(dimensions=[0, 1], bounds=[(-5, 5), (-5, 5)], resolution=100)
'''

'''
if __name__ == "__main__":

    # Create an instance of the Rastrigin objective function
    rastrigin = Ackley(dimensionality=2, search_space_bounds=[(-5, 5), (-5, 5)])

    # Visualize the objective function in 2D
    rastrigin.visualize(dimensions=[0, 1], bounds=[(-5, 5), (-5, 5)], resolution=100)

    import timeit
    import numpy as np

    # Create an instance of the Ackley class
    ackley = Ackley(dimensionality=2, search_space_bounds=[(-5, 5), (-5, 5)])

    # Measure the execution time of evaluate_a
    time_a = timeit.timeit(lambda: ackley.evaluate(np.random.uniform(-5, 5, 2)), number=10000)

    # Measure the execution time of evaluate_b
    time_b = timeit.timeit(lambda: ackley.evaluate_b(np.random.uniform(-5, 5, 2)), number=10000)

    print(f"Execution time of evaluate: {time_a} seconds")
    print(f"Execution time of evaluate_b: {time_b} seconds")
'''

'''
if __name__ == "__main__":

    # Create an instance of the Griewank objective function
    griewank = Griewank(dimensionality=2, search_space_bounds=[(-5, 5), (-5, 5)])

    # Visualize the objective function in 2D
    griewank.visualize(dimensions=[0, 1], bounds=[(-5, 5), (-5, 5)], resolution=100)
'''


if __name__ == "__main__":

    # Create an instance of the Michalewicz objective function
    michalewicz = Michalewicz(dimensionality=2, search_space_bounds=[(0, np.pi), (0, np.pi)])

    # Visualize the objective function in 2D
    michalewicz.visualize(dimensions=[0, 1], bounds=[(0, np.pi), (0, np.pi)], resolution=100)

    '''

    # Create an instance of the Styblinski-Tang objective function
    styblinski_tang = StyblinskiTang(dimensionality=2, search_space_bounds=[(-5, 5), (-5, 5)])

    # Visualize the objective function in 2D
    styblinski_tang.visualize(dimensions=[0, 1], bounds=[(-5, 5), (-5, 5)], resolution=100)

    '''