
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_PARAMETERS = {'m': 10}

DEFAULT_SETTINGS = DefaultSettings(dimensionality=2,
                                   optimal_solution=None,
                                   optimal_solution_position=None,
                                   search_space_bounds=[(0, np.pi), (0, np.pi)],
                                   clamping_method='random')


class Michalewicz(of.ObjectiveFunction):

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
        Evaluates the Michalewicz function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Michalewicz function value at the given position.
        """
        return -np.sum(np.sin(position) * (np.sin((np.arange(1, len(position) + 1) * position**2) / np.pi))**(2 * self.parameters['m']))
