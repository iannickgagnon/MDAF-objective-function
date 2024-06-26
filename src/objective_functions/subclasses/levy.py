
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
