
# External libraries
import numpy as np


class DefaultSettings:
    """
    A class representing the default settings for the objective function.

    Args:
        dimensionality (int): The dimensionality of the problem.
        optimal_solution (float): The optimal solution value.
        optimal_solution_position (np.ndarray): The position of the optimal solution.
        search_space_bounds (np.ndarray): The bounds of the search space.

    Attributes:
        Same as the arguments.
    """

    def __init__(self, 
                 dimensionality: int,
                 optimal_solution: float, 
                 optimal_solution_position: np.ndarray, 
                 search_space_bounds: np.ndarray):
        self.dimensionality = dimensionality
        self.optimal_solution = optimal_solution
        self.optimal_solution_position = optimal_solution_position
        self.search_space_bounds = search_space_bounds

    # Implement dictionary-like behavior
    def __iter__(self):
        return iter(self.__dict__)
    
    def __next__(self):
        return next(self.__dict__)
    
    def __getitem__(self, key):
        return getattr(self, key)
    