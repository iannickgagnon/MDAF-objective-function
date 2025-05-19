# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_PARAMETERS = {'A': np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]]), 
                      'm': 5,
                      'c': np.array([1, 2, 5, 2, 3])}

DEFAULT_SETTINGS = DefaultSettings(dimensionality=2,
                                   optimal_solution=None,
                                   optimal_solution_position=None,
                                   search_space_bounds=np.array([[0, 10], [0, 10]]))


class Langermann(of.ObjectiveFunction):
    """
    Langermann objective function implementation.

    This class implements the Langermann function, a multimodal benchmark function commonly used for testing optimization algorithms.
    The function features multiple local minima, making it suitable for evaluating the performance of global optimization techniques.

    Attributes:
        parameters (dict): Dictionary containing the parameters 'A', 'm', and 'c' for the Langermann function.
        settings (DefaultSettings): Object containing the search space bounds and other configuration values.

    Methods:
        __init__(parameters: dict = {}, settings: DefaultSettings = {}):
            Initializes the function with provided parameters and settings, applying defaults if necessary.
        evaluate(position: np.ndarray) -> float:
            Computes the Langermann function value at the specified position.

    Reference:
        https://github.com/iannickgagnon/MDAF_objective_functions/wiki/Langermann
    """

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
        c = self.parameters['c']
        
        result = 0.0
        for i in range(m):
            xi = position - A[i]
            sum_xi_squared = np.sum(xi**2)
            term = c[i] * np.exp(-sum_xi_squared / np.pi) * np.cos(np.pi * sum_xi_squared)
            result += term
            
        return -result
