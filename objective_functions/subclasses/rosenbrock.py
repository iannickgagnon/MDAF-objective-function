
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(dimensionality = 2, 
                                   optimal_solution = 0.0, 
                                   optimal_solution_position = np.array([1.0, 1.0]),
                                   search_space_bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]]), 
                                   clamping_method = 'random')


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
        return (1 - position[0]) ** 2 + 100 * (position[1] - position[0] ** 2) ** 2
    