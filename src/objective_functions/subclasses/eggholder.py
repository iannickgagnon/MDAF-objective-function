
# External libraries
import numpy as np

# Internal classes
from .. import objective_function as of
from ..default_settings import DefaultSettings

DEFAULT_SETTINGS = DefaultSettings(dimensionality=2,
                                   optimal_solution=-959.6407,
                                   optimal_solution_position=np.array([512, 404.2319]),
                                   search_space_bounds=np.array([[-512, 512], [-512, 512]]))


class Eggholder(of.ObjectiveFunction):

    @of.constructor
    def __init__(self,
                 parameters: dict = {},
                 settings: DefaultSettings = {}):

        # Validate default settings
        self.validate_settings(settings, DEFAULT_SETTINGS)

    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the Eggholder function at the given position.

        Args:
            position (np.ndarray): The position to evaluate.

        Returns:
            float: The Eggholder function value at the given position.
        """

        x1 = position[0]
        x2 = position[1]

        term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x1 / 2 + (x2 + 47))))
        term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))

        return term1 + term2
