# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_PARAMETERS = {"m": 10}

DEFAULT_SETTINGS = DefaultSettings(
    ndim=2,
    optimal_solution=None,
    optimal_solution_position=None,
    search_space_bounds=[(0, np.pi), (0, np.pi)],
)


class Michalewicz(of.ObjectiveFunction):
    """
    Michalewicz objective function implementation.

    This class implements the Michalewicz function, a benchmark function commonly used to evaluate optimization algorithms.
    The function is known for its steep valleys and ridges, making it difficult for optimization methods to find the global minimum.

    Attributes:
        parameters (dict): Dictionary containing the parameter 'm' for the Michalewicz function.
        settings (DefaultSettings): Object containing the search space and other configuration values.

    Methods:
        __init__(parameters: dict = {}, settings: DefaultSettings = {}):
            Initializes the function with given parameters and settings, applying defaults if needed.
        evaluate(position: np.ndarray) -> float:
            Computes the function value at the given position.

    Reference:
        https://github.com/iannickgagnon/MDAF_objective_functions/wiki/Michalewicz
    """

    @of.constructor
    def __init__(self, parameters: dict = {}, settings: DefaultSettings = {}):

        # Validate the parameters and apply default values if necessary
        self.validate_parameters(parameters, DEFAULT_PARAMETERS)

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Michalewicz function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Michalewicz function value at the given position.
        """
        return -np.sum(
            np.sin(position)
            * (np.sin((np.arange(1, len(position) + 1) * position**2) / np.pi))
            ** (2 * self.parameters["m"])
        )
