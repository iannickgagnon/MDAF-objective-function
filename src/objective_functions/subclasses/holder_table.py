
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_PARAMETERS = {'A': 10, 'B': 0.2, 'C': np.pi}

DEFAULT_SETTINGS = DefaultSettings(dimensionality=2,
                                   optimal_solution=-19.2085,
                                   optimal_solution_position=np.array([8.05502, 9.66459]),
                                   search_space_bounds=np.array([[-10, 10], [-10, 10]]))


class HolderTable(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 parameters: dict = {},
                 settings: DefaultSettings = {}):

        # Validate the parameters and apply default values if necessary
        self.validate_parameters(parameters, DEFAULT_PARAMETERS)

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Holder Table function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Holder Table function value at the given position.
        """

        x = position[:, 0]
        y = position[:, 1]

        return -np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - (np.sqrt(x**2 + y**2) / np.pi))))
