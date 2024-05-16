
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(dimensionality = 2, 
                                   optimal_solution = 0.0, 
                                   optimal_solution_position = np.array([1.0, 1.0]),
                                   search_space_bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]]))


class Rosenbrock(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 settings: DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Rosenbrock function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Rosenbrock function value at the given position.
        """

        if position.ndim == 1:
            position = position.reshape(1, -1)

        x = position[:, 0]
        y = position[:, 1]

        return (1 - x) ** 2 + 100 * (y - y ** 2) ** 2
    