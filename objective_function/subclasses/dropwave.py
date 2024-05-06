
# External libraries
import numpy as np

# Internal classes
from .. import objective_function

class DropWave(objective_function.ObjectiveFunction):

    def __init__(self, 
                 parameters: dict = {},
                 dimensionality: int = 2, 
                 optimal_solution: float = -1.0,
                 optimal_solution_position: np.ndarray = np.array([0, 0]),
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
        Evaluates the Dropwave function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Dropwave function value at the given position.
        """
        x = position[0]
        y = position[1]
        numerator = 1 + np.cos(12 * np.sqrt(x**2 + y**2))
        denominator = 0.5 * (x**2 + y**2) + 2
        return -numerator / denominator
