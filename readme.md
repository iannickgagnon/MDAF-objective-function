![](./doc/img/logo_top.png)
![](./doc/img/logo_bottom.png)

The Metaheuristics Design And Analysis Framework (MDAF) is an open-source collection of Python modules designed specifically for the metaheuristics research community. It provides off-the-shelf tools that promote and facilitate rigorous scientific research based on firmly established best practices.

# The Objective Function Evaluation Module

The `MDAF-objective-function` module  is a specialized tool designed to streamline the process of defining and evaluating objective functions. It offers a low-code template that simplifies the setup and execution of complex objective functions, allowing researchers to concentrate more on refining their algorithms and less on the intricacies of coding.

## Version

This library uses calendar versioning (CalVer) (`MAJOR.YEAR.NUMBER`) where : 

- `MAJOR`: This represents a major release.
- `YEAR`: The year of the release.
- `NUMBER`: The release number for the given year.

Calendar versioning was preferred over semantic versioning (SemVer) to facilitate interpretation in journal articles.

## How to cite

**APA Style**:
```
Gagnon, I. (YEAR). Metaheuristics Design and Analysis Framework (MDAF), Version MAJOR.YEAR.NUMBER [Software]. Retrieved from https://github.com/iannickgagnon/MDAF-objective-function.
```

**MLA Style**: 
```
Gagnon, Iannick. Metaheuristics Design And Analysis Framework. Version MAJOR.YEAR.NUMBER. YEAR. Software. Accessed DAY-MONTH-YEAR. https://github.com/iannickgagnon/MDAF-objective-function.
```

**Chicago Style**:
```
Gagnon, Iannick. YEAR. Metaheuristics Design And Analysis Framework (MDAF), Version MAJOR.YEAR.NUMBER. Software. https://github.com/iannickgagnon/MDAF-objective-function.
```

Replace `YEAR` with the year of the version used, `MAJOR.YEAR.NUMBER` with the full version and `DAY-MONTH-YEAR` with the date you accessed the software online (i.e., the date at which you cloned the repo).

## Features

- **Automated accounting**: The number times a function is evaluated is automatically incremented and stored (`ObjectiveFunction.nb_calls`).
- **Automatic differentiation**: Built-in numerical first- and second-order differentiation (`ObjectiveFunction.compute_first_derivative()` and `ObjectiveFunction.compute_second_derivative()`).
- **Parallel evaluation**: Built-in support for parallel evaluation (`ObjectiveFunction.parallel_evaluate()`).
- **Profiling**: Built-in profiling of the objective function's caller function (`ObjectiveFunction.profile_evaluate()`).
- **Shifting**: Apply shifts to evaluate robustness to positional bias (`ObjectiveFunction.apply_shift()`).
- **Rotating**: Apply rotations to evaluate robustness to linkage (`ObjectiveFunction.apply_rotation()`).
- **Noisy evaluation**: Apply noise to evaluate robustness to noisy environments (`ObjectiveFunction.apply_noise()`).
- **Timing**: Built-in timing for performance optimization and comparison (`ObjectiveFunction.time()`).
- **Visualization**: Dynamic visualization in 2D or 3D (`ObjectiveFunction.visualize()`).

## How to install

Install Python's package manager if you don't have it already:

* https://pip.pypa.io/en/stable/installation/

Run the following command:

```bash
pip install MDAF-objective-function
```

Use the following command to install this library:

## How to clone

Install the open source git version control software if you don't have it already: 

* https://git-scm.com/downloads

Clone this repository to your local machine using:

```bash
git clone https://github.com/iannickgagnon/MDAF-objective-function.git
```

If you aren't familiar with git, you may wish to consult the following online e-book for free:

* https://git-scm.com/book/en/v2 

I have also greatly benefited from the following official cheat sheet from GitHub:
* https://education.github.com/git-cheat-sheet-education.pdf

## Built-in objective functions

## Future work

- **Multiple objectives**: Derive a `MultiobjectiveFunction` class from `ObjectiveFunction`. See Issue https://github.com/iannickgagnon/MDAF-objective-function/issues/9.
- **Enhance timing**: Enable timing for the `ObjectiveFunction.parallel_evaluate()` method. See Issue https://github.com/iannickgagnon/MDAF-objective-function/issues/11.

## Contact

**Please note that I am not looking for prospective students. I sincerely appreciate your messages, but I cannot respond to them because of other priorities.**

While I cannot guarantee a timely response, I invite you to contact me if you have any comments : iannick.gagnon@etsmtl.ca. Also, feel free to add me on LinkedIn : https://www.linkedin.com/in/iannick-gagnon-3304311a6/.
