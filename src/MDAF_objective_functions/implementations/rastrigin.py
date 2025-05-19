
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_PARAMETERS = {'A': 10}

DEFAULT_SETTINGS = DefaultSettings(dimensionality = 2, 
                                   optimal_solution = 0.0, 
                                   optimal_solution_position = np.zeros(2), 
                                   search_space_bounds = np.array([[-5.15, 5.12], [-5.12, 5.12]]))


class Rastrigin(of.ObjectiveFunction):
    """
    Rastrigin objective function implementation.

    This class implements the Rastrigin function, a standard benchmark for optimization algorithms.
    The Rastrigin function is highly multimodal with a large number of local minima, making it a challenging
    test case for global optimization methods.

    Attributes:
        parameters (dict): Dictionary containing the parameter 'A' for the Rastrigin function.
        settings (DefaultSettings): Object containing the search space and other configuration values.

    Methods:
        __init__(parameters: dict = {}, settings: DefaultSettings = {}):
            Initializes the function with given parameters and settings, applying defaults if needed.
        evaluate(position: np.ndarray) -> float:
            Computes the function value at the given position.
    
    Reference:
        https://github.com/iannickgagnon/MDAF_objective_functions/wiki/Rastrigin
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
        Evaluates the Rastrigin function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Rastrigin function value at the given position.
        """    
        n = len(position)
        A = self.parameters.get('A', 10)
        return A * n + np.sum(position**2 - A * np.cos(2 * np.pi * position))
    