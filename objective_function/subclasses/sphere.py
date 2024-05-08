
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of

DEFAULT_SETTINGS = of.DefaultSettings(dimensionality = 2, 
                                      optimal_solution = 0.0, 
                                      optimal_solution_position = np.zeros(2), 
                                      search_space_bounds = np.array([(-5, 5), (-5, 5)]), 
                                      clamping_method = 'random')


class Sphere(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 settings: of.DefaultSettings = {}):

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
