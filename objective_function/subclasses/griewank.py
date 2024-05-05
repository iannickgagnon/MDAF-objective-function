
# External libraries
import numpy as np

# Internal classes
from .. import objective_function


class Griewank(objective_function.ObjectiveFunction):

    def __init__(self, 
                 parameters: dict = {},
                 dimensionality: int = 2, 
                 optimal_solution: float = 0.0,
                 optimal_solution_position: np.ndarray = np.zeros(2),
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
        Evaluates the Griewank function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Griewank function value at the given position.
        """
        
        term1 = np.sum(position**2) / 4000
        term2 = np.prod(np.cos(position / np.sqrt(np.arange(1, len(position) + 1))))

        return term1 - term2 + 1
