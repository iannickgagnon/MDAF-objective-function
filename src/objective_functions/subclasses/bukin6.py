
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(dimensionality = 2, 
                                   optimal_solution = 0.0, 
                                   optimal_solution_position = np.array([-10, 1]), 
                                   search_space_bounds = np.array([(-15, -5), (-3, 3)]))


class Bukin6(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 settings: DefaultSettings = {}):

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

        if position.ndim == 1:
            position = position.reshape(1, -1)

        x = position[:, 0]
        y = position[:, 1]

        return 100 * np.sqrt(np.abs(y - 0.01 * x**2)) + 0.01 * np.abs(x + 10)
