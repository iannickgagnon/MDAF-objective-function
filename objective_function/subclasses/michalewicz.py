
# External libraries
import numpy as np

# Internal classes
from .. import objective_function


class Michalewicz(objective_function.ObjectiveFunction):

    def __init__(self, 
                 parameters: dict = {},
                 dimensionality: int = 2, 
                 optimal_solution: float = None,
                 optimal_solution_position: np.ndarray = None,
                 search_space_bounds: dict = None,
                 clamping_method: str = None):
        
        # Validate parameters
        if 'm' not in parameters:
            print("\033[93mWarning: The parameter 'm' is not set. The default value of 10 will be used.\033[0m")
            parameters['m'] = 10

        super().__init__(parameters, 
                         dimensionality, 
                         optimal_solution, 
                         optimal_solution_position, 
                         search_space_bounds, 
                         clamping_method)
        
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Michalewicz function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Michalewicz function value at the given position.
        """
        return -np.sum(np.sin(position) * (np.sin((np.arange(1, len(position) + 1) * position**2) / np.pi))**(2 * self.parameters['m']))
