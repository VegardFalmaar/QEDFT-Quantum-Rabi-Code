QEDFT-Quantum-Rabi
====

This repository contains numerics and plots for the QEDFT Quantum Rabi paper.
(Specific links and IDs will be added.)

The functionality for most of the main computations is contained in the
`QuantumRabi` class in `quantum_rabi.py`.


Dependencies
====

- [qmodel](https://github.com/magmage/qmodel) (v. 0.2.2)
- numpy
- scipy
- numba
- matplotlib
- pytest


Tests
====

The file `test_results_of_paper.py` contains tests of various equations and
inequalities of the paper. They are set up using `pytest`, and every
proposition is tested with a combination of different parameters.

The tests pass with the current value of the tolerance when the `gtol` is
reduced to `1e-6` in the minimization of the Legendre transform in `dft.py` in
`qmodel`.

Run the tests with
```sh
pytest test_results_of_paper.py
```

To run only one of the tests in the file use
```sh
pytest test_results_of_paper.py::name_of_specific_test_function
```


Plots
====

The file `plot_config.py` contains a lot of configuration options that are
chosen to suit the configuration of the paper, such as figure size, fonts and
fontsizes. It also includes some utility functions for enabling LaTeX fonts,
setting the axis labels and titles with the correct fontsizes, resizing figures
etc.

Plots are saved in the `plots/` directory. There is a rough correspondence
between the names of the Python source files and the names of the saved PDFs.
