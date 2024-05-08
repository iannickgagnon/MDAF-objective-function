
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of

DEFAULT_SETTINGS = of.DefaultSettings(dimensionality=2,
                                      optimal_solution=None,
                                      optimal_solution_position=None,
                                      search_space_bounds=[(-5, 5), (-5, 5)],
                                      clamping_method='random')


class StyblinskiTang(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 settings: of.DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Styblinski-Tang function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Styblinski-Tang function value at the given position.
        """
        return np.sum(position**4 - 16 * position**2 + 5 * position) / 2
