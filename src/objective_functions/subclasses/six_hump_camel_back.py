
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(dimensionality=2,
                                   optimal_solution=-1.0316,
                                   optimal_solution_position=np.array([[0.0898, -0.7126], [-0.0898, 0.7126]]),
                                   search_space_bounds=np.array([(-5, 5), (-5, 5)]))


class SixHumpCamelBack(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 settings: DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Six-Hump Camelback function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Six-Hump Camelback function value at the given position.
        """

        if position.ndim == 1:
            position = position.reshape(1, -1)

        x = position[:, 0]
        y = position[:, 1]

        return (4 - 2.1 * x**2 + (x**4) / 3) * x**2 + x * y + (-4 + 4 * y**2) * y**2
