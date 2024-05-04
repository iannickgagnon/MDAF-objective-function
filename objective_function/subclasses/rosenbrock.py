
# External libraries
import numpy as np

# Internal classes
from .. import objective_function


class Rosenbrock(objective_function.ObjectiveFunction):

    def __init__(self, 
                 parameters: dict = None,
                 dimensionality: int = 2, 
                 optimal_solution: float = 0.0,
                 optimal_solution_position: np.ndarray = np.zeros(2),
                 search_space_bounds: dict = None,
                 clamping_method: str = None):
        
        super().__init__(parameters, dimensionality, optimal_solution, optimal_solution_position, search_space_bounds, clamping_method)
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Rosenbrock function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Rosenbrock function value at the given position.
        """
        return (1 - position[0]) ** 2 + 100 * (position[1] - position[0] ** 2) ** 2