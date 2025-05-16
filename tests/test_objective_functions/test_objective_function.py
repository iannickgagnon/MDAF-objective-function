import os
import pytest
import numpy as np
from unittest.mock import MagicMock
from MDAF_objective_functions.objective_function import ObjectiveFunction, count_calls, constructor

# Fixtures for creating instances of ObjectiveFunction

@pytest.fixture
def mock_objective_function():
    # Mock the abstract methods to instantiate ObjectiveFunction
    class ConcreteObjectiveFunction(ObjectiveFunction):
        def evaluate(self, position):
            return np.sum(position**2)
    
    obj_fn = ConcreteObjectiveFunction()
    obj_fn.dimensionality = 2
    obj_fn.optimal_solution_position = np.array([1, 1])
    obj_fn.search_space_bounds = np.array([[0, 2], [0, 2]])
    return obj_fn

@pytest.fixture
def invalid_objective_function():
    # Object with mismatched dimensions for testing initialization failure
    class InvalidObjectiveFunction(ObjectiveFunction):
        def evaluate(self, position):
            return np.sum(position**2)
    
    obj_fn = InvalidObjectiveFunction()
    obj_fn.dimensionality = 3
    obj_fn.optimal_solution_position = np.array([1, 1])
    obj_fn.search_space_bounds = np.array([[0, 2], [0, 2]])
    return obj_fn

def test_valid_initialization(mock_objective_function):
    assert mock_objective_function.nb_calls == 0
    assert isinstance(mock_objective_function.shift, np.ndarray)

def test_invalid_initialization(invalid_objective_function):
    with pytest.raises(ValueError):
        invalid_objective_function.__init__()

def test_count_calls_decorator(mock_objective_function):
    decorated = count_calls(mock_objective_function.evaluate)
    mock_objective_function.evaluate = decorated
    assert mock_objective_function.nb_calls == 0
    mock_objective_function.evaluate(np.array([1, 1]))
    assert mock_objective_function.nb_calls == 1

def test_constructor_decorator():
    # Demonstrating the use of the constructor decorator
    class TestClass:
        def __init__(self, x):
            self.x = x
    
    class ChildClass(TestClass):
        @constructor
        def __init__(self, x):
            self.x = x + 1
    
    obj = ChildClass(1)
    assert obj.x == 2  # Constructor logic
    assert isinstance(obj, TestClass)  # Super constructor logic

def test_parallel_evaluate(mock_objective_function, mocker):
    mocker.patch('os.path.exists', return_value=True)
    mocker.patch('objective_functions.tmp.decoupled_evaluate.evaluate', return_value=1)
    result = mock_objective_function.parallel_evaluate(np.array([[1, 1], [2, 2]]))
    assert np.array_equal(result, np.array([1, 1]))

def test_visualize(mock_objective_function, mocker):
    mocker.patch('matplotlib.pyplot.show')  # Mock plt.show to not actually display plot during tests
    mock_objective_function.visualize()
    # No exception means pass, you might want to check for calls or handle plt objects.

def test_compute_first_derivative(mock_objective_function, mocker):
    mock_objective_function.first_derivative = MagicMock(return_value=np.array([1, 1]))
    result = mock_objective_function.compute_first_derivative(np.array([1, 1]))
    assert np.array_equal(result, np.array([1, 1]))

def test_save_and_load(mock_objective_function, mocker, tmp_path):
    # Use tmp_path fixture provided by pytest for handling file operations
    file_path = tmp_path / "obj_fn.pkl"
    mocker.patch('builtins.open', mocker.mock_open())
    mocker.patch('pickle.dump')
    mocker.patch('pickle.load', return_value=mock_objective_function)
    
    mock_objective_function.save(str(file_path))
    loaded_fn = ObjectiveFunction.load(str(file_path))
    
    assert isinstance(loaded_fn, ObjectiveFunction)
    