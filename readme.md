![](./doc/img/logo_top.png)
![](./doc/img/logo_bottom.png)

The Metaheuristics Design And Analysis Framework (MDAF) is an open-source collection of Python modules designed specifically for the metaheuristics research community. It provides off-the-shelf tools that promote and facilitate rigorous scientific research based on firmly established best practices.

# The Objective Function Evaluation Module

The `MDAF-objective-function` module  is a specialized tool designed to streamline the process of defining and evaluating objective functions. It offers a low-code template that simplifies the setup and execution of complex objective functions, allowing researchers to concentrate more on refining their algorithms and less on the intricacies of coding.

## Features

- **Automated accounting**: The number times a function is evaluated is automatically incremented and stored (`objective_function.nb_calls`).
- **Automatic differentiation**: Built-in numerical first- and second-order differentiation (`objective_function.compute_first_derivative()` and `objective_function.compute_second_derivative()`).
- **Parallel evaluation**: Built-in support for parallel evaluation (`objective_function.parallel_evaluate()`).
- **Shifting**: Apply shifts to evaluate robustness to positional bias (`objective_function.apply_shift()`).
- **Rotating**: Apply rotations to evaluate robustness to linkage (`objective_function.apply_rotation()`).
- **Noisy evaluation**: Apply noise to evaluate robustness to noisy environments (`objective_function.apply_noise()`).
- **Visualization**: Dynamic visualization in 2D or 3D (`objective_function.visualize()`).

## Built-in objective functions

## Installation

Install the open source git version control software if you don't have it already: 

* https://git-scm.com/downloads

Clone this repository to your local machine using:

```bash
git clone https://github.com/iannickgagnon/MDAF-objective-function.git
```

If you aren't familiar with git, you may wish to consult the following online e-book for free:

* https://git-scm.com/book/en/v2 

## Known limitations

## Future work

## Contact