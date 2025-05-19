
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(dimensionality=2,
                                   optimal_solution=0,
                                   optimal_solution_position=np.array([1, 1]), 
                                   search_space_bounds=np.array([[-10, 10], [-10, 10]]))


class Levy(of.ObjectiveFunction):
    """
    Levy objective function implementation.

    This class implements the Levy function, a standard benchmark for testing optimization algorithms.
    The Levy function is known for its complex landscape with many local minima, making it difficult
    for optimization methods to find the global minimum.

    Attributes:
        settings (DefaultSettings): Object containing the search space bounds, dimensionality, and other configuration values.

    Methods:
        __init__(settings: DefaultSettings = {}):
            Initializes the function with the provided settings, applying defaults if necessary.
        evaluate(position: np.ndarray) -> float:
            Computes the Levy function value at the specified position.

    Reference:
        https://github.com/iannickgagnon/MDAF_objective_functions/wiki/Levy
    """

    @of.constructor
    def __init__(self,
                 settings: DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Levy function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Levy function value at the given position.
        """

        if position.ndim == 1:
            position = position.reshape(1, -1)

        w = 1 + (position - 1) / 4
        
        t1 = np.sum(np.sin(np.pi * w[:, 1]) ** 2)
        t2 = np.sum((w[:, :-1] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[:, :-1] + 1) ** 2), axis=1)
        t3 = np.sum((w[:, -1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[:, -1]) ** 2))

        return t1 + t2 + t3
