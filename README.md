# SparseBayes.jl

[![Build Status](https://travis-ci.org/irustandi/SparseBayes.jl.svg?branch=master)](https://travis-ci.org/irustandi/SparseBayes.jl) [![Coverage Status](https://coveralls.io/repos/irustandi/SparseBayes.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/irustandi/SparseBayes.jl?branch=master)

A Julia package for sparse Bayesian learning
--------------------------------------------

This package is based on the [SparseBayes package for Matlab] (http://www.miketipping.com/sparsebayes.htm#software) written by Mike Tipping.

## Installation

SparseBayes.jl is written for Julia version 0.4. To install, run the following command inside a Julia session:

```julia
julia> Pkg.add("SparseBayes")
```

SparseBayes.jl requires the [Optim.jl] (https://github.com/JuliaOpt/Optim.jl) package.

## Examples

Some examples of using the sparse Bayesian learning package is available in the examples/ folder.