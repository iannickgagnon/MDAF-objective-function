# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(
    dimensionality=2,
    optimal_solution=None,
    optimal_solution_position=None,
    search_space_bounds=[(-5, 5), (-5, 5)],
)


class StyblinskiTang(of.ObjectiveFunction):
    """
    Styblinski-Tang objective function implementation.

    This class implements the Styblinski-Tang function, a widely used benchmark in global optimization.
    The Styblinski-Tang function is multimodal and non-convex, presenting numerous local minima, which
    makes it a challenging test case for optimization algorithms.

    Attributes:
        settings (DefaultSettings): Object containing the search space and other configuration values.

    Methods:
        __init__(settings: DefaultSettings = {}):
            Initializes the function with the provided settings, applying defaults if necessary.
        evaluate(position: np.ndarray) -> float:
            Computes the Styblinski-Tang function value at the given position.

    Reference:
        https://github.com/iannickgagnon/MDAF_objective_functions/wiki/Styblinski%E2%80%90Tang
    """

    @of.constructor
    def __init__(self, settings: DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Styblinski-Tang function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Styblinski-Tang function value at the given position.
        """
        return np.sum(position**4 - 16 * position**2 + 5 * position) / 2
