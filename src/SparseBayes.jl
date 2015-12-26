module SparseBayes

# package code goes here
export
AbstractRVMModel,
AbstractRVMTrainingOptions,
RVMBernoulliModel,
RVMGaussianModel,
RVMBernoulliTrainingOptions,
RVMGaussianTrainingOptions,
fit!,
predict,
sigmoid,
kernelLinear,
kernelGaussian

include("kernels.jl")
include("RVM.jl")

end # module
