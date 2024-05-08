
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of

DEFAULT_SETTINGS = of.DefaultSettings(dimensionality = 2, 
                                      optimal_solution = -1.0, 
                                      optimal_solution_position = np.zeros(2), 
                                      search_space_bounds = np.array([(-5.12, 5.12), (-5.12, 5.12)]), 
                                      clamping_method = 'random')


class DropWave(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 parameters: dict = {},
                 settings: of.DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Dropwave function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Dropwave function value at the given position.
        """
        x, y = position
        numerator = 1 + np.cos(12 * np.sqrt(x**2 + y**2))
        denominator = 0.5 * (x**2 + y**2) + 2
        return -numerator / denominator