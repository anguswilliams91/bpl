# bpl

[![Build Status](https://travis-ci.org/anguswilliams91/bpl.svg?branch=master)](https://travis-ci.org/anguswilliams91/bpl)
[![codecov](https://codecov.io/gh/anguswilliams91/bpl/branch/master/graph/badge.svg)](https://codecov.io/gh/anguswilliams91/bpl)

`bpl` is a python 3 library for fitting Bayesian versions of the Dixon \& Coles (1997) model to data.
It uses the `stan` library to fit models to data.  

 ## Installation

You will need a working C++ compiler.
If you are using anaconda, you can install gcc with  

```bash
conda install gcc
``` 

Since the project uses a `pyproject.toml`, an up-to-date version of `pip` is needed as well (>=18.0).
Once these dependencies are present, you can install from source using `pip`:

```bash
pip install https://github.com/anguswilliams91/bpl/archive/master.zip
```
