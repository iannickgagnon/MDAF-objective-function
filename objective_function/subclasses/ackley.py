
# External libraries
import numpy as np

# Internal classes
from .. import objective_function

class Ackley(objective_function.ObjectiveFunction):

    def __init__(self, 
                 parameters: dict = {},
                 dimensionality: int = None,
                 clamping_method: str = None):
        
        # Configuration
        if not dimensionality:
            print("\033[93mWARNING: The input parameter 'dimensionality' is not set. The default value of 2 is used instead.\033[0m")
            dimensionality = 10

        if 'A' not in parameters:
            print("\033[93mWARNING: The parameter 'A' is not set. The default value of 10 is used instead.\033[0m")
            parameters['A'] = 10

        if 'B' not in parameters:
            print("\033[93mWARNING: The parameter 'B' is not set. The default value of 0.2 is used instead.\033[0m")
            parameters['B'] = 0.2

        if 'C' not in parameters:
            print("\033[93mWARNING: The parameter 'C' is not set. The default value of 2π is used instead.\033[0m")
            parameters['C'] = 2 * np.pi

        # Adjust the optimal solution and search space bounds
        optimal_solution = 0.0
        optimal_solution_position = np.zeros(dimensionality)
        search_space_bounds = np.array([[-32.768, 32.768]] * dimensionality)

        super().__init__(parameters = parameters, 
                         dimensionality = dimensionality, 
                         optimal_solution_fitness = optimal_solution, 
                         optimal_solution_position = optimal_solution_position, 
                         search_space_bounds = search_space_bounds, 
                         clamping_method = clamping_method)
        
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