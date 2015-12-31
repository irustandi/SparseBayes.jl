module TestProcessRVMAction

using SparseBayes
using Base.Test

alpha = 1.
s = 0.1
factor = 0.01
addValue = (s * s) / factor

@test isinf(SparseBayes.processRVMAction(SparseBayes.Delete, alpha, s, factor))
@test_approx_eq SparseBayes.processRVMAction(SparseBayes.Hold, alpha, s, factor) alpha
@test_approx_eq SparseBayes.processRVMAction(SparseBayes.Add, alpha, s, factor) addValue

end
