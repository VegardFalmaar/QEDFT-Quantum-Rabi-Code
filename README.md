QEDFT-Quantum-Rabi
====

Numerics and plots for the QEDFT Quantum Rabi paper.


Tests
====

The file `test_results_of_paper.py` contains tests of various equations and
inequalities of the paper. They are set up using `pytest`, and every equation
proposition is tested with a combination of different parameters.

To run the tests in all files whole name start with `test_`, simply run
```sh
pytest
```
in this directory.

To run only the tests in `test_results_of_paper.py`, use
```sh
pytest test_results_of_paper.py
```

To run only one test in the file use
```sh
pytest test_results_of_paper.py::name_of_specific_test_function
```


Plots
====

The file `plot_config.py` contains a lot of configuration options that are
chosen to look suit the configuration of the paper, such as figure size, fonts
and fontsizes. It also includes some utility functions for enabling LaTeX
fonts, setting the axis labels and titles with the correct fontsizes, resizing
figures etc.

See the `plot_functional_in_lambda.py` for an example of the use.
