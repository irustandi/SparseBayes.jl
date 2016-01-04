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

module TestCalculateFactorQuantities

using SparseBayes
using Base.Test

alpha = 1.
S = 0.5
Q = 0.1

s, q, factor = SparseBayes.calculateFactorQuantities(alpha, S, Q)
multTerm = alpha / (alpha - S)
sRef = multTerm * S
qRef = multTerm * Q
factorRef = qRef * qRef - sRef
@test_approx_eq s sRef
@test_approx_eq q qRef
@test_approx_eq factor factorRef

factorRef = Q * Q - S
s, q, factor = SparseBayes.calculateFactorQuantities(Inf, S, Q)
@test_approx_eq s S
@test_approx_eq q Q
@test_approx_eq factor factorRef

end
