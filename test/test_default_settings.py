import pytest
import numpy as np
from src.MDAF_benchmarks.default_settings import DefaultSettings

def test_initialization():
    """
    Test the initialization of the DefaultSettings class by verifying that 
    the DefaultSettings object is correctly instantiated with the provided 
    parameters.
    """
    
    # Test parameters
    ndim = 3
    optimal_solution = 0.0
    optimal_solution_position = np.array([0.1, 0.2, 0.3])
    search_space_bounds = np.array([[0, 1], [0, 1], [0, 1]])

    # Create an instance of DefaultSettings
    settings = DefaultSettings(ndim, optimal_solution, optimal_solution_position, search_space_bounds)

    # Check that the attributes are set correctly
    assert settings.ndim == ndim
    assert settings.optimal_solution == optimal_solution
    np.testing.assert_array_equal(settings.optimal_solution_position, optimal_solution_position)
    np.testing.assert_array_equal(settings.search_space_bounds, search_space_bounds)

def test_dict_like_getitem():
    """
    Test the dictionary-like item access of the DefaultSettings class by verifying 
    that values can be retrieved using the __getitem__ method with string keys.
    """

    # Create a test instance of DefaultSettings
    settings = DefaultSettings(
        ndim=2,
        optimal_solution=1.5,
        optimal_solution_position=np.array([0.5, 0.5]),
        search_space_bounds=np.array([[0, 1], [0, 1]])
    )

    # Test dictionary-like access
    assert settings['ndim'] == 2
    assert settings['optimal_solution'] == 1.5
    np.testing.assert_array_equal(settings['optimal_solution_position'], np.array([0.5, 0.5]))
    np.testing.assert_array_equal(settings['search_space_bounds'], np.array([[0, 1], [0, 1]]))

def test_dict_like_iter():
    """
    Test the dictionary-like iteration of the DefaultSettings class by verifying
    that the keys can be iterated over and match the expected keys.
    """
    
    # Create a test instance of DefaultSettings
    settings = DefaultSettings(
        ndim=2,
        optimal_solution=1.5,
        optimal_solution_position=np.array([0.5, 0.5]),
        search_space_bounds=np.array([[0, 1], [0, 1]])
    )

    # Extract acutal keys from the settings object
    keys = list(iter(settings))

    # Define the expected keys
    expected_keys = ['ndim', 
                     'optimal_solution', 
                     'optimal_solution_position', 
                     'search_space_bounds']
    
    # Check that the keys match the expected keys ignoring order with set comparison
    assert set(keys) == set(expected_keys)

def test_next_raises_stop_iteration():
    """
    Test that the StopIteration exception is raised when the iterator is exhausted by iterating 
    over the settings object and then calling next() on the iterator.
    """

    # Create a test instance of DefaultSettings
    settings = DefaultSettings(
        ndim=1,
        optimal_solution=0.0,
        optimal_solution_position=np.array([0.0]),
        search_space_bounds=np.array([[0, 1]])
    )
    
    # Create an iterator from the settings object
    it = iter(settings)
    
    # Iterate over the settings object
    for _ in settings: 
        next(it)
    
    # Check that StopIteration is raised by exhausting the iterator
    with pytest.raises(StopIteration):
        next(it)
