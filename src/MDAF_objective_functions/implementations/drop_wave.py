# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(
    ndim=2,
    optimal_solution=-1.0,
    optimal_solution_position=np.zeros(2),
    search_space_bounds=np.array([(-5.12, 5.12), (-5.12, 5.12)]),
)


class DropWave(of.ObjectiveFunction):
    """
    DropWave objective function implementation.

    This class implements the DropWave function, a standard benchmark for testing optimization algorithms.
    The DropWave function is characterized by its steep valleys and multiple local minima, making it a
    challenging landscape for optimization techniques.

    Attributes:
        settings (DefaultSettings): Object containing the search space bounds, optimal solution, and other configuration values.

    Methods:
        __init__(settings: DefaultSettings = {}):
            Initializes the function with the provided settings, applying defaults if necessary.
        evaluate(position: np.ndarray) -> float:
            Computes the DropWave function value at the specified position.

    Reference:
        https://github.com/iannickgagnon/MDAF_objective_functions/wiki/Drop%E2%80%90Wave
    """

    @of.constructor
    def __init__(self, settings: DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Dropwave function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Dropwave function value at the given position.
        """

        if position.ndim == 1:
            position = position.reshape(1, -1)

        x = position[:, 0]
        y = position[:, 1]

        numerator = 1 + np.cos(12 * np.sqrt(x**2 + y**2))
        denominator = 0.5 * (x**2 + y**2) + 2

        return -numerator / denominator
