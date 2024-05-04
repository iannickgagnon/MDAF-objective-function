
from objective_function.subclasses.sphere import Sphere

if __name__ == "__main__":

    # Create an instance of the Sphere objective function
    sphere = Sphere(dimensionality=2, search_space_bounds=[(-5, 5), (-5, 5)])

    # Visualize the objective function in 2D
    sphere.visualize(dimensions=[0, 1], bounds=[(-5, 5), (-5, 5)], resolution=100)