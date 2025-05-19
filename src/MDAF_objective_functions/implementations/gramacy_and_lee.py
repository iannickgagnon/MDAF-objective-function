# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(
    ndim=1,
    optimal_solution=-0.869011135,
    optimal_solution_position=np.array([0.5485634]),
    search_space_bounds=np.array([[0.5, 2.5]]),
)


class GramacyAndLee(of.ObjectiveFunction):
    """
    Gramacy & Lee objective function implementation.

    This class implements the Gramacy & Lee function, a benchmark function commonly used for testing
    optimization algorithms. The function is known for its non-convexity and multiple local minima,
    providing a challenging landscape for optimization.

    Attributes:
        settings (DefaultSettings): Object containing the search space bounds, optimal solution, and other configuration values.

    Methods:
        __init__(settings: DefaultSettings = {}):
            Initializes the function with the provided settings, applying defaults if necessary.
        evaluate(position: np.ndarray) -> float:
            Computes the function value at the specified position.

    Reference:
        https://github.com/iannickgagnon/MDAF_objective_functions/wiki/Gramacy-and-Lee
    """

    @of.constructor
    def __init__(self, settings: DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Gramacy & Lee function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Gramacy & Lee function value at the given position.
        """
        return np.sin(10 * np.pi * position) / (2 * position) + (position - 1) ** 4
