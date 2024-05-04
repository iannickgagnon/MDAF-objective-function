
# External libraries
import numpy as np

# Internal classes
from .. import objective_function


class Rastrigin(objective_function.ObjectiveFunction):

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

        super().__init__(parameters, 
                         dimensionality, 
                         optimal_solution, 
                         optimal_solution_position, 
                         search_space_bounds, 
                         clamping_method)
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Rastrigin function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Rastrigin function value at the given position.
        """
        n = len(position)
        return self.parameters['A'] * n + np.sum(position**2 - self.parameters['A'] * np.cos(2 * np.pi * position))