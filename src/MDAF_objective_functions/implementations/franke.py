# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(
    dimensionality=2,
    optimal_solution=-1.2185807,
    optimal_solution_position=np.array([0.2, 0.2]),
    search_space_bounds=np.array([(0, 1), (0, 1)]),
)


class Franke(of.ObjectiveFunction):
    """
    Franke objective function implementation.

    This class implements the Franke function, a commonly used benchmark for testing optimization algorithms,
    especially in the context of surface fitting and interpolation. The function features multiple local minima
    and a known global minimum, providing a challenging landscape for optimization methods.

    Attributes:
        settings (DefaultSettings): Object containing the search space bounds, optimal solution, and other configuration values.

    Methods:
        __init__(settings: DefaultSettings = {}):
            Initializes the function with provided or default settings, validating configuration.
        evaluate(position: np.ndarray) -> float:
            Computes the Franke function value at the specified position.

    Reference:
        https://github.com/iannickgagnon/MDAF_objective_functions/wiki/Franke
    """

    @of.constructor
    def __init__(self, settings: DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates Franke's function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Franke's function value at the given position.
        """

        if position.ndim == 1:
            position = position.reshape(1, -1)

        x = position[:, 0]
        y = position[:, 1]

        term1 = 0.75 * np.exp(-((9 * x - 2) ** 2) / 4 - (9 * y - 2) ** 2 / 4)
        term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49 - (9 * y + 1) / 10)
        term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4 - (9 * y - 3) ** 2 / 4)
        term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)

        return -(term1 + term2 + term3 + term4)
