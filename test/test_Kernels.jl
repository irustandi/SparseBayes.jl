module TestKernelLinear

using SparseBayes
using Base.Test

X1 = ones(2, 1)
X2 = ones(3, 1)

K = kernelLinear(X1, X2)
nRows, nCols = size(K)

@test size(K, 1) == size(X1, 1)
@test size(K, 2) == size(X2, 1)

end

module TestKernelGaussian

using SparseBayes
using Base.Test

sigma = 0.5
X1 = ones(2, 1)
X2 = ones(3, 1)

K = kernelGaussian(X1, X2, sigma)
nRows, nCols = size(K)

@test size(K, 1) == size(X1, 1)
@test size(K, 2) == size(X2, 1)

end
