
# External libraries
import numpy as np

# Internal classes
from .. import objective_function


class Bulkin6(objective_function.ObjectiveFunction):

    def __init__(self, 
                 parameters: dict = {},
                 dimensionality: int = 2, 
                 optimal_solution: float = 0,
                 optimal_solution_position: np.ndarray = np.array([-10, 1]),
                 search_space_bounds: dict = None,
                 clamping_method: str = None):
        
        super().__init__(parameters, 
                         dimensionality, 
                         optimal_solution, 
                         optimal_solution_position, 
                         search_space_bounds, 
                         clamping_method)
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Bulkin no. 6 function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Bulkin no. 6 function value at the given position.
        """
        return 100 * np.sqrt(np.abs(position[1] - 0.01 * position[0]**2)) + 0.01 * np.abs(position[0] + 10)
