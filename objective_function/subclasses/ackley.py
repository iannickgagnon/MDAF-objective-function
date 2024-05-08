
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of

DEFAULT_PARAMETERS = {'A': 10, 'B': 0.2, 'C': 2 * np.pi}

DEFAULT_SETTINGS = of.DefaultSettings(dimensionality = 2, 
                                      optimal_solution = 0.0, 
                                      optimal_solution_position = np.zeros(2), 
                                      search_space_bounds = np.array([[-32.768, 32.768], [-32.768, 32.768]]), 
                                      clamping_method = 'random')
    
    
class Ackley(of.ObjectiveFunction):

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
        Evaluates the Ackley function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Ackley function value at the given position.
        """
        
        n = len(position)
        
        A = self.parameters['A']
        B = self.parameters['B']
        C = self.parameters['C']
        
        term1 = -A * np.exp(-B * np.sqrt(np.sum(position**2) / n))
        term2 = -np.exp(np.sum(np.cos(C * position)) / n)
        return term1 + term2 + A + np.exp(1)
    