
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_PARAMETERS = {'A': np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]]), 
                      'm': 5}

DEFAULT_SETTINGS = DefaultSettings(dimensionality=2,
                                   optimal_solution=None,
                                   optimal_solution_position=None,
                                   search_space_bounds=np.array([[0, 10], [0, 10]]))


class Langermann(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 parameters: dict = {},
                 settings: DefaultSettings = {}):

        # Validate the parameters and apply default values if necessary
        self.validate_parameters(parameters, DEFAULT_PARAMETERS)

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Langermann function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Langermann function value at the given position.
        """
        
        m = self.parameters['m']
        A = self.parameters['A']
        
        xi = position - A
        sum_xi_squared = np.sum(xi**2, axis=1)
        sum_xi = np.sum(xi, axis=1)
    
        return -np.sum(np.exp(-sum_xi_squared / np.pi) * np.cos(np.pi * sum_xi))