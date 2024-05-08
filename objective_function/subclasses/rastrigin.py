
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of

DEFAULT_PARAMETERS = {'A': 10}

DEFAULT_SETTINGS = of.DefaultSettings(dimensionality = 2, 
                                      optimal_solution = 0.0, 
                                      optimal_solution_position = np.zeros(2), 
                                      search_space_bounds = np.array([[-5.15, 5.12], [-5.12, 5.12]]), 
                                      clamping_method = 'random')


class Rastrigin(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 parameters: dict = {},
                 settings: of.DefaultSettings = {}):

        # Validate the parameters and apply default values if necessary
        self.validate_parameters(parameters, DEFAULT_PARAMETERS)

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Rastrigin function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Rastrigin function value at the given position.
        """
        n = len(position)
        A = self.parameters.get('A', 10)
        return A * n + np.sum(position**2 - A * np.cos(2 * np.pi * position))
    
