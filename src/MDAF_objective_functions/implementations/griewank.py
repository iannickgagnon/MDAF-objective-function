# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(
    dimensionality=2,
    optimal_solution=0.0,
    optimal_solution_position=np.zeros(2),
    search_space_bounds=np.array([[-32.768, 32.768], [-32.768, 32.768]]),
)


class Griewank(of.ObjectiveFunction):
    """
    Griewank objective function implementation.

    This class implements the Griewank function, a commonly used benchmark for testing optimization algorithms.
    The function is characterized by many widespread local minima, making it difficult for optimization methods
    to find the global minimum.

    Attributes:
        settings (DefaultSettings): Object containing the search space bounds, dimensionality, and other configuration values.

    Methods:
        __init__(settings: DefaultSettings = {}):
            Initializes the function with the provided settings, applying defaults if necessary.
        evaluate(position: np.ndarray) -> float:
            Computes the Griewank function value at the specified position.

    Reference:
        https://github.com/iannickgagnon/MDAF_objective_functions/wiki/Griewank
    """

    @of.constructor
    def __init__(self, settings: DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Griewank function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Griewank function value at the given position.
        """

        term1 = np.sum(position**2) / 4000
        term2 = np.prod(np.cos(position / np.sqrt(np.arange(1, len(position) + 1))))

        return term1 - term2 + 1
