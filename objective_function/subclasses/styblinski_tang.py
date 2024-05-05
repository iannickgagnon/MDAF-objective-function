
# External libraries
import numpy as np

# Internal classes
from .. import objective_function


class StyblinskiTang(objective_function.ObjectiveFunction):

    def __init__(self, 
                 parameters: dict = {},
                 dimensionality: int = 2, 
                 optimal_solution: float = -39.16599 * 2,
                 optimal_solution_position: np.ndarray = np.array([-2.903534, -2.903534]),
                 search_space_bounds: dict = None,
                 clamping_method: str = None):
        
        # Validate parameters
        if 'a' not in parameters:
            print("\033[93mWarning: The parameter 'a' is not set. The default value of 2 will be used.\033[0m")
            parameters['a'] = 2

        super().__init__(parameters, 
                         dimensionality, 
                         optimal_solution, 
                         optimal_solution_position, 
                         search_space_bounds, 
                         clamping_method)
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Styblinski-Tang function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Styblinski-Tang function value at the given position.
        """
        return np.sum(position**4 - 16 * position**2 + 5 * position) / 2
