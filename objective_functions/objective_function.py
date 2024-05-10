
# External libraries
import os
import re
import pickle
import inspect
import numpy as np
from copy import deepcopy
from numbers import Number
from typing import Callable 
from typing import Iterable
import matplotlib.pyplot as plt
from autograd import grad, hessian
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed

# Internal constants
LEFT_CLICK = 1
RIGHT_CLICK = 3

# Internal paths
DECOUPLED_FUNCTION_PATH = './objective_functions/tmp/decoupled_evaluate.py'


def count_calls(foo: Callable) -> Callable:
    """
    Decorator that counts the number of calls to a method.

    Args:
        foo (Callable): The function to count the number of calls to.

    Returns:
        Callable: The decorated function.
    """
    def wrapper(self, *args, **kwargs):

        # Increment the number of calls
        self.nb_calls += 1

        # Call the original function
        return foo(self, *args, **kwargs)
    
    return wrapper


def constructor(foo: Callable):
    """
    Calls the super constructor after executing the subclass constructor.

    Args:
        foo (Callable): The subclass constructor to be executed before calling the super constructor.

    Returns:
        Callable: The wrapper function that executes the given function and calls the super constructor.

    """
    def wrapper(self, **kwargs):

        # Call the subclass constructor
        foo(self, **kwargs)

        # Call the super constructor
        super(self.__class__, self).__init__(**kwargs)

    return wrapper


class ObjectiveFunction(ABC):

    parameters: dict = {}

    def __init__(self):
        
        # Validate the size of the optimal solution position
        if self.optimal_solution_position is not None and len(self.optimal_solution_position) != self.dimensionality:
                raise ValueError("The size of the optimal solution position must match the dimensionality of the objective function.")

        # Validate the size of the search space bounds
        if len(self.search_space_bounds) != self.dimensionality:
            raise ValueError("The size of the search space bounds must match the dimensionality of the objective function.")

        # Initialize shift
        self.shift: np.ndarray = np.zeros(self.dimensionality)

        # Initialize the number of objective function evaluations
        self.nb_calls: int = 0

        # Compute the first and second derivatives of the objective function's evaluation method
        self.first_derivative = grad(self.evaluate)
        self.second_derivative = hessian(self.evaluate)


    def validate_parameters(self, parameters: dict, default_parameters: dict):
        """
        Validates the parameters of the objective function.

        Args:
            parameters (dict): The parameters to validate.
            default_parameters (dict): The default parameters of the objective function.

        Returns:
            dict: The validated parameters.
        """

        # Check if the number of parameters exceeds the number of default parameters
        if len(parameters) > len(default_parameters):
            raise ValueError("The number of parameters exceeds the number of default parameters.")

        # Make sure that all the provided parameters are also in the default parameters
        for parameter_name in parameters:
            if parameter_name not in default_parameters:
                raise ValueError(f"'{parameter_name}' is not a valid parameter for this objective function. Valid parameters are {default_parameters.keys()}.")

        # Store the parameters and set default values as required
        for parameter_name in default_parameters:
            if parameter_name in parameters:
                self.parameters[parameter_name] = parameters[parameter_name]
            else:
                default_value = default_parameters[parameter_name]
                print(f"\033[93mWARNING: The '{parameter_name}' parameter is not set. The default value of {default_value} is used.\033[0m")
                self.parameters[parameter_name] = default_value
       

    def validate_settings(self, settings: dict, default_settings: dict):
        """
        Validates the settings of the objective function.

        Args:
            settings (dict): The settings to validate.
            default_settings (dict): The default settings of the objective function.

        Returns:
            dict: The validated settings.
        """

        # Store the settings and set default values as required
        for setting_name in default_settings:
            if setting_name in settings:
                self.__dict__[setting_name] = settings[setting_name]
            else:
                default_value = default_settings[setting_name]
                print(f"\033[93mWARNING: The '{setting_name}' setting is not set. The default value of {default_value} is used.\033[0m")
                self.__dict__[setting_name] = default_value
    
    @count_calls
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
    
    
    def parallel_evaluate(self, positions: np.ndarray, max_workers: int = None) -> np.ndarray:
        """
        Evaluates multiple positions in parallel using ProcessPoolExecutor.

        Args:
            positions (np.ndarray): An array of positions to evaluate. Each row corresponds to a position.
            max_workers (int): The maximum number of processes that can be used to execute the given calls.

        Returns:
            np.ndarray: An array of objective function values corresponding to the input positions.
        """

        try:

            # Export the evaluate function as a top-level function in a separate file
            self.__decouple_evaluate()

            # Import the decoupled evaluate function
            if os.path.exists(DECOUPLED_FUNCTION_PATH):
                try:
                    from objective_functions.tmp.decoupled_evaluate import evaluate as decoupled_evaluate   # type: ignore
                except Exception as e:
                    raise Exception(f"Failed to import the decoupled evaluate method with traceback: {e}")

        except Exception as e:

            # Delete the decoupled evaluate function file
            self.__delete_decoupled_evaluate()

            raise Exception(f"Failed to decouple the evaluate method with traceback: {e}")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:

            # Submit all evaluations to the executor
            future_to_position = {executor.submit(decoupled_evaluate, pos): pos for pos in positions}

            # Collect results as they are completed
            results = np.zeros(len(positions))
            for future in as_completed(future_to_position):
                pos = future_to_position[future]
                try:
                    result = future.result()
                except Exception as exc:
                    print(f'{pos} generated an exception: {exc}')
                else:
                    index = np.where((positions == pos).all(axis=1))[0][0]
                    results[index] = result
        
        # Delete the decoupled evaluate function file
        self.__delete_decoupled_evaluate()

        return results


    def visualize(self, 
                  dimensions: Iterable[int] = (0, 1), 
                  plot_bounds: Iterable[Iterable[Number]] = None, 
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
        
        if len(dimensions) not in (1, 2):
            raise ValueError("The number of dimensions to visualize must be 2 or 3.")
        
        # Adjust the plot bounds
        if plot_bounds and len(plot_bounds) != len(dimensions):
            raise ValueError("The number of bounds must match the number of dimensions.")
        else:
            plot_bounds = self.search_space_bounds
        
        if len(dimensions) == 1:

            # Create the figure and axis
            fig, ax = plt.subplots(figsize=(8, 6))

            # Define the grid for the single dimension
            x = np.linspace(plot_bounds[0][0], plot_bounds[0][1], resolution)

            # Vectorized evaluation of the objective function
            Z = self.evaluate(x.reshape(-1, 1))  # Reshape for function evaluation, assuming 'evaluate' can handle an array

            # Draw the line plot
            ax.plot(x, Z, label=f'Objective Function along X{dimensions[0]}')
            ax.set_xlabel(f'X{dimensions[0]}')
            ax.set_ylabel('Fitness')
            ax.set_title('1D Visualization of Objective Function')
            ax.legend()

            plt.show()

        elif len(dimensions) == 2:

            # Create the figure and axes
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))

            # Define the grid
            x_min, x_max = plot_bounds[0]
            y_min, y_max = plot_bounds[1]
            x = np.linspace(x_min, x_max, resolution)
            y = np.linspace(y_min, y_max, resolution)
            X, Y = np.meshgrid(x, y)

            # Vectorized evaluation of the objective function
            positions = np.vstack([X.ravel(), Y.ravel()]).T
            Z = np.array([self.evaluate(position) for position in positions]).reshape(X.shape)

            # Draw the contour plot with level curves
            levels = np.linspace(np.min(Z), np.max(Z), num=min(resolution // 10, 10))
            cs = axs[0].contourf(X, Y, Z, levels=levels, cmap='jet')
            axs[0].contour(cs, colors='k', linewidths=1.0)
            axs[0].set_xlabel(f"X{dimensions[0]}")
            axs[0].set_ylabel(f"X{dimensions[1]}")
            axs[0].set_title("2D Contour Plot")

            # Show the optimal solution on the contour plot
            if self.optimal_solution_position is not None:
                axs[0].scatter(self.optimal_solution_position[dimensions[0]],
                            self.optimal_solution_position[dimensions[1]],
                            color='yellow', marker='*', edgecolor='black', s=200)
            
            def on_contour_click(event) -> None:
                """
                Event handler for mouse clicks on the contour plot. Adds a red sphere marker on 
                the 3D plot for a left-click and removes the last marker for a right-click.

                Args:
                    event (MouseEvent): The mouse click event.

                Returns:
                    None
                """

                if event.button == LEFT_CLICK:

                    # Get the clicked coordinates
                    x, y = event.xdata, event.ydata

                    # This is a workaround to avoid a 'bug' in matplotlib where a left-click event is always triggered around (0, 0)
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
     
            # Show the optimal solution on the surface plot
            if self.optimal_solution_position is not None:
                axs[1].scatter(self.optimal_solution_position[dimensions[0]], 
                               self.optimal_solution_position[dimensions[1]], 
                               self.optimal_solution, 
                               color='yellow', 
                               marker='*', 
                               edgecolor='black', 
                               s=200)

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

        # Check if the search space bounds have been defined
        if not self.search_space_bounds:
            raise ValueError("No constraints on the search space have been defined for this objective function.")

        # Check if the solution satisfies the constraints for each dimension
        return np.any((position < self.search_space_bounds[:, 0]) | (position > self.search_space_bounds[:, 1]))
    

    def apply_shift(self, shift: np.ndarray) -> None:
        """
        Shifts the given position by the given shift vector.

        Args:
            position (np.ndarray): The position to shift.
            shift (np.ndarray): The shift vector.

        Returns:
            np.ndarray: The shifted position.
        """
        
        # Shift the optimal solution position
        if self.optimal_solution_position is not None:
            self.optimal_solution_position += shift

        # Copy the original evaluate method
        evaluate_copy = deepcopy(self.evaluate)

        # Define a new evaluate method that shifts the position before evaluating
        def shifted_evaluate(position: np.ndarray) -> np.ndarray:
            return evaluate_copy(position - shift)
        
        # Replace the original evaluate method with the shifted one
        self.evaluate = shifted_evaluate
    
    
    def apply_noise(self, noisy_foo: Callable) -> None:

         # Copy the original evaluate method
        evaluate_copy = deepcopy(self.evaluate)

        # Define a new evaluate method that shifts the position before evaluating
        def noisy_evaluate(position: np.ndarray) -> np.ndarray:
            return evaluate_copy(position) + noisy_foo()
        
        # Replace the original evaluate method with the shifted one
        self.evaluate = noisy_evaluate


    def save(self, path: str) -> None:
        """
        Saves the objective function to a file.

        Args:
            path (str): The path to save the objective function to.
        """
        with open(path, 'wb') as file:
            pickle.dump(self, file)
            print("\033[92mObjectiveFunction state saved in {path}\033[0m")

            
    @staticmethod
    def load(path: str):
        """
        Deserialize the state of the objective function from a file.

        Args:
            filename (str): The path to the file from which the object state will be loaded.

        Returns:
            ObjectiveFunction: The deserialized objective function object.
        """
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            print("\033[92mObjectiveFunction state loaded from {path}\033[0m")
            return obj


    def __decouple_evaluate(self):
        """
        Decouples the evaluate method from the objective function instance.
        """

        # Get the cleartext code of the evaluate method
        cleartext_code = inspect.getsource(self.evaluate)

        # Remove docstrings
        cleartext_code = re.sub(r'\"\"\".*?\"\"\"', '', cleartext_code, flags=re.DOTALL)

        # Split the code into lines
        raw_lines = cleartext_code.split('\n')

        # Remove empty lines
        code_lines = [line for line in raw_lines if line.strip()]

        # Count the number of leading spaces of the first line
        leading_spaces = len(code_lines[0]) - len(code_lines[0].lstrip())

        # Remove the first leading spaces from all the lines
        code_lines = [line[leading_spaces:] for line in code_lines]

        # Combine the code lines into a single string
        reassembled_code = '\n'.join(code_lines)

        # Remove references to self
        reassembled_code = reassembled_code.replace('self,', '')

        # Save the decoupled evaluate method to a file
        with open(DECOUPLED_FUNCTION_PATH, 'w') as file_write:
            
            with open(__file__, 'r') as file_read:

                # Import all the libraries used in the objective_function.py file
                for line in file_read.readlines():
                    if 'def' in line:
                        break
                    elif 'import' in line:
                        file_write.write(f'{line.strip()}\n')

            # Write the decoupled evaluate method
            file_write.write(reassembled_code)

    @staticmethod
    def __delete_decoupled_evaluate() -> None:
        """
        Deletes the decoupled evaluate method file.
        """
        if os.path.exists(DECOUPLED_FUNCTION_PATH):
            os.remove(DECOUPLED_FUNCTION_PATH)
