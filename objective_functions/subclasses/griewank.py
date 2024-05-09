
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(dimensionality= 2,
                                   optimal_solution = 0.0, 
                                   optimal_solution_position = np.zeros(2), 
                                   search_space_bounds = np.array([[-32.768, 32.768], [-32.768, 32.768]]), 
                                   clamping_method = 'random')


class Griewank(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 settings: DefaultSettings = {}):
        
        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Griewank function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Griewank function value at the given position.
        """
        
        term1 = np.sum(position**2) / 4000
        term2 = np.prod(np.cos(position / np.sqrt(np.arange(1, len(position) + 1))))

        return term1 - term2 + 1
