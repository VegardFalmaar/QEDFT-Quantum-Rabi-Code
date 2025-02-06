QEDFT-Quantum-Rabi
====

This repository contains code and results for a QEDFT treatment of the Quantum
Rabi model and related models. (Links to relevant publications will be added.)


Dependencies
====

- [qmodel](https://github.com/magmage/qmodel) (v. 0.2.2)
- numpy
- scipy
- numba
- matplotlib
- pytest


Structure
====

The functionality for most of the main computations is contained in the
`QuantumRabi` class in `quantum_rabi.py`.

The file `plot_config.py` contains configuration options that are chosen to
suit the configuration of the papers, such as figure size, fonts and fontsizes.
It also includes some utility functions for enabling LaTeX fonts, setting the
axis labels and titles with the correct fontsizes, resizing figures etc.

Plots are saved in the `plots/` directory. There is a rough correspondence
between the names of the Python source files and the names of the saved PDFs.


Tests
====

The file `test_results_of_paper.py` contains tests of various equations and
inequalities of the paper. They are set up using `pytest`, and the propositions
are tested with a combination of different parameters.

Run the tests with
```sh
pytest test_results_of_paper.py
```

To run only one of the tests in the file use
```sh
pytest test_results_of_paper.py::name_of_specific_test_function
```
