import numpy as np
from numbers import Number
from typing import Iterable, Union
import matplotlib.pyplot as plt
from autograd import grad, hessian
from abc import ABC, abstractmethod

# Internal constants
LEFT_CLICK = 1
RIGHT_CLICK = 3


class ObjectiveFunction(ABC):

    def __init__(self,  
                 parameters: dict = None,
                 dimensionality: int = None, 
                 optimal_solution_fitness: float = None,
                 optimal_solution_position: np.ndarray = None,
                 search_space_bounds: np.ndarray = None,
                 clamping_method: str = None):
        
        # Validate the size of the optimal solution position
        assert len(optimal_solution_position) == dimensionality, "The size of the optimal solution position must match the dimensionality of the objective function."
        self.parameters = parameters
        self.dimensionality = dimensionality
        self.optimal_solution_fitness = optimal_solution_fitness
        self.optimal_solution_position = optimal_solution_position
        self.search_space_bounds = search_space_bounds
        self.clamping_method = clamping_method
        self.first_derivative = grad(self.evaluate)
        self.second_derivative = hessian(self.evaluate)

    @abstractmethod
    def evaluate(self, position: np.ndarray) -> float:
        """
        Evaluates the objective function at the given position.

        Args:
            solution (np.ndarray): The solution to evaluate.

        Returns:
            float: The objective function value at the given solution.
        """
        pass
    
    def visualize(self, 
                  dimensions: Iterable[int], 
                  bounds: Iterable[Iterable[Number]], 
                  resolution: int = 100):
        """
        Visualizes the objective function in 2D or 3D.

        Args:
            dimensions (Iterable[int]): The dimensions to visualize. Must be an iterable of length 2 or 3.
            bounds (Iterable[Iterable[Number]]): The bounds of the visualization. Must be an iterable of length 2 or 3.
            resolution (int): The resolution of the visualization grid.

        Raises:
            AssertionError: If the number of dimensions is not 2 or 3.
            AssertionError: If the number of bounds does not match the number of dimensions.
        """
        
        assert len(dimensions) in [1, 2], "The number of dimensions to visualize must be 2 or 3."
        assert len(bounds) == len(dimensions), "The number of bounds must match the number of dimensions."
        
        if len(dimensions) == 1:

            pass

        else:

            # 2D contour visualization
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Define the grid
            x = np.linspace(bounds[0][0], bounds[0][1], resolution)
            y = np.linspace(bounds[1][0], bounds[1][1], resolution)
            X, Y = np.meshgrid(x, y)
            
            # Evaluate the objective function at each grid point
            Z = np.zeros((resolution, resolution))
            for i in range(resolution):
                for j in range(resolution):
                    Z[i, j] = self.evaluate(np.array([X[i, j], Y[i, j]]))

            # Draw the contour plot with level curves
            levels = np.linspace(np.min(Z), np.max(Z), resolution // 10)
            cs = axs[0].contourf(X, Y, Z, levels=levels, cmap='jet')
            axs[0].contour(cs, colors='k', linewidths=1.0)
            axs[0].set_xlabel(f"X{dimensions[0]}")
            axs[0].set_ylabel(f"X{dimensions[1]}")
            axs[0].set_title("Contour Plot")
            axs[0].contour(cs, colors='k')

            def on_contour_click(event):
    
                if event.button == LEFT_CLICK:

                    # Get the clicked coordinates
                    x, y = event.xdata, event.ydata

                    # This is a workaround to avoid a bug in matplotlib where a left-click even is always triggered around (0, 0)
                    if x is None or x is None or (np.abs(x) < 0.1 and np.abs(y) < 0.1):
                        return

                    # Create a sphere marker on the 3D plot for a left-click
                    axs[1].scatter(x, y, self.evaluate(np.array([x, y])), color='red', marker='o', s=100)

                if event.button == RIGHT_CLICK:
                    
                    # Remove the last sphere marker on the 3D plot for a right-click
                    if len(axs[1].collections) > 1:
                        axs[1].collections[-1].remove()
                    
                plt.draw()

            # Register the on_contour_click function as the callback for the contour plot
            cs.figure.canvas.mpl_connect('button_press_event', on_contour_click)

            # 3D surface visualization
            axs[1].axis('off')
            axs[1] = fig.add_subplot(122, projection='3d')
            axs[1].plot_surface(X, Y, Z, cmap='jet', alpha=0.5)
            axs[1].set_xlabel(f"X{dimensions[0]}")
            axs[1].set_ylabel(f"X{dimensions[1]}")
            axs[1].set_zlabel(f"f(X{dimensions[0]}, X{dimensions[1]})")
            axs[1].set_title("Surface Plot")
     
            # Adjust the layout
            plt.tight_layout()

            plt.show()


    def compute_first_derivative(self, position: np.ndarray) -> np.ndarray:
        """
        Computes the first derivative of the objective function at the given position.

        Args:
            position (np.ndarray): The position at which to compute the first derivative.

        Returns:
            np.ndarray: The first derivative of the objective function at the given position.
        """
        return self.first_derivative(position)
        

    def compute_second_derivative(self, position: np.ndarray) -> np.ndarray:
        """
        Computes the second derivative of the objective function at the given position.

        Args:
            position (np.ndarray): The position at which to compute the second derivative.

        Returns:
            np.ndarray: The second derivative of the objective function at the given position.
        """
        return self.second_derivative(position)


    def check_constraints(self, position: np.ndarray) -> bool:
        """
        Checks if the given solution satisfies the constraints defined by the search space bounds.

        Args:
            solution (np.ndarray): The solution to check.

        Returns:
            bool: True if the solution satisfies the constraints, False otherwise.
        
        Raises:
            ValueError: If no constraints on the search space have been defined for this objective function.
        """

        assert self.search_space_bounds, "No constraints on the search space have been defined for this objective function."

        # Check if the solution satisfies the constraints for each dimension
        return np.any((position < self.search_space_bounds[:, 0]) | (position > self.search_space_bounds[:, 1]))
    