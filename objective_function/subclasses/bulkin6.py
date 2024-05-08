
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of

DEFAULT_SETTINGS = of.DefaultSettings(dimensionality = 2, 
                                      optimal_solution = 0.0, 
                                      optimal_solution_position = np.array([-10, 1]), 
                                      search_space_bounds = np.array([(-15, -5), (-3, 3)]), 
                                      clamping_method = 'random')


class Bulkin6(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 settings: of.DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Bulkin no. 6 function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Bulkin no. 6 function value at the given position.
        """
        return 100 * np.sqrt(np.abs(position[1] - 0.01 * position[0]**2)) + 0.01 * np.abs(position[0] + 10)
