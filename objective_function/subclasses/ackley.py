
# External libraries
import numpy as np

# Internal classes
from .. import objective_function
import timeit


class Ackley(objective_function.ObjectiveFunction):

    def __init__(self, 
                 parameters: dict = {},
                 dimensionality: int = 2, 
                 optimal_solution: float = 0.0,
                 optimal_solution_position: np.ndarray = np.zeros(2),
                 search_space_bounds: dict = None,
                 clamping_method: str = None):
        
        # Validate parameters
        if 'A' not in parameters:
            print("\033[93mWarning: The parameter 'A' is not set. The default value of 10 will be used.\033[0m")
            parameters['A'] = 10

        if 'B' not in parameters:
            print("\033[93mWarning: The parameter 'B' is not set. The default value of 0.2 will be used.\033[0m")
            parameters['B'] = 0.2

        if 'C' not in parameters:
            print("\033[93mWarning: The parameter 'C' is not set. The default value of 2 * pi will be used.\033[0m")
            parameters['C'] = 2 * np.pi

        super().__init__(parameters, 
                         dimensionality, 
                         optimal_solution, 
                         optimal_solution_position, 
                         search_space_bounds, 
                         clamping_method)
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Ackley function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Ackley function value at the given position.
        """
        
        n = len(position)
        
        A = self.parameters.get('A')
        B = self.parameters.get('B')
        C = self.parameters.get('C')
        
        term1 = -A * np.exp(-B * np.sqrt(np.sum(position**2) / n))
        term2 = -np.exp(np.sum(np.cos(C * position)) / n)
        return term1 + term2 + A + np.exp(1)