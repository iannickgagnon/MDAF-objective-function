
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(dimensionality = 2, 
                                   optimal_solution = 0.0, 
                                   optimal_solution_position = np.zeros(2), 
                                   search_space_bounds = np.array([(-5, 5), (-5, 5)]))


class Sphere(of.ObjectiveFunction):
    """
    Sphere objective function implementation.
    
    This class implements the Sphere function, a fundamental benchmark in optimization. The Sphere function is
    characterized by its simple, convex, and continuous landscape, making it a standard test case for evaluating
    the performance of optimization algorithms.
    
    Attributes:
        settings (DefaultSettings): Object containing the search space and other configuration values.
    
    Methods:
        __init__(settings: DefaultSettings = {}):
            Initializes the function with the provided settings, applying defaults if necessary.
        evaluate(position: np.ndarray) -> float:
            Computes the Sphere function value at the given position.
    """
    
    @of.constructor
    def __init__(self,
                 settings: DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Sphere function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Sphere function value at the given position.
        """
        return np.sum(position ** 2)
