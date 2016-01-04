using Optim;
@enum RVMAction Hold Reestimate Add Delete

abstract AbstractRVMModel

type RVMGaussianModel <: AbstractRVMModel
    kernelFun::Function
    X::Matrix
    alpha::Vector
    w::Vector
    beta::Float64

    RVMGaussianModel(kernelFun) = new(kernelFun, Array{Float64, 2}(0, 0), zeros(0), zeros(0), NaN)
end

type RVMBernoulliModel <: AbstractRVMModel
    kernelFun::Function
    X::Matrix
    alpha::Vector
    w::Vector
    beta::Vector

    RVMBernoulliModel(kernelFun) = new(kernelFun, Array{Float64, 2}(0, 0), zeros(0), zeros(0), zeros(0))    
end

abstract AbstractRVMTrainingOptions

type RVMGaussianTrainingOptions <: AbstractRVMTrainingOptions
    logAlphaTol::Float64
    betaUpdateStart::Int64
    betaUpdateFreq::Int64
    priorityDeletion::Bool
    priorityAddition::Bool

    RVMGaussianTrainingOptions(logAlphaTol = 1e-3, betaUpdateStart = 10, betaUpdateFreq = 5, priorityDeletion = true, priorityAddition = false) = new(logAlphaTol, betaUpdateStart, betaUpdateFreq, priorityDeletion, priorityAddition)
end
    
type RVMBernoulliTrainingOptions <: AbstractRVMTrainingOptions
    logAlphaTol::Float64
    priorityDeletion::Bool
    priorityAddition::Bool

    RVMBernoulliTrainingOptions(logAlphaTol = 1e-3, priorityDeletion = true, priorityAddition = false) = new(logAlphaTol, priorityDeletion, priorityAddition)
end

using Optim;

function sigmoid(x::Vector)
    f = 1 ./ (1 + exp(-x));
end

function rvmBernoulliPosterior(params::Vector, Phi::Matrix, t::Vector, alpha::Vector)
    y = sigmoid(Phi * params);
    A = diagm(alpha);

    f = sum(-t .* log(y) - (1-t) .* log(1-y)) + 0.5 .* transpose(params) * A * params
    return f[1]
end

function rvmBernoulliPosterior_grad!(params::Vector, storage::Vector, Phi::Matrix, t::Vector, alpha::Vector)
    y = sigmoid(Phi * params)
    e = t - y

    storage[:] = -transpose(Phi) * e + alpha .* params
end

function rvmBernoulliPosterior_hess!(params::Vector, storage::Matrix, Phi::Matrix, t::Vector, alpha::Vector)
    y = sigmoid(Phi * params)
    beta = y .* (1 - y)
    Phi_B = Phi .* (beta * ones(1, size(Phi, 2)))
    A = diagm(alpha)
    storage[:] = (transpose(Phi_B) * Phi + A)
end

function getSubset(alpha::Vector, Phi::Matrix)
    indices = !isinf(alpha)
    Phi_sub = Phi[:, indices]
    alpha_sub = alpha[indices]
    A = diagm(alpha_sub)

    return indices, Phi_sub, alpha_sub, A
end

function calculateRVMPosterior!(mdl::RVMGaussianModel, Phi::Matrix, t::Vector, Phi_t::Matrix, Phi_t_Target::Vector, Phi_t_Phi_diag::Vector)
    indices, Phi_sub, alpha_sub, A = getSubset(mdl.alpha, Phi)
    
    Phi_sub_t = transpose(Phi_sub)
    Hessian = A + mdl.beta * Phi_sub_t * Phi_sub
    U = chol(Hessian)
    Ui = inv(U)
    # posterior covariance
    Sigma::Matrix = Ui * transpose(Ui)

    Sigma_Phit::Matrix = Sigma * Phi_sub_t
    Sigma_Phit_Target::Vector = Sigma_Phit * t
    # posterior mean
    Mu::Vector = mdl.beta .* Sigma_Phit_Target

    y = Phi_sub * Mu
    e = t - y
    e_sq::Float64 = dot(e, e)

    N, M_all = size(Phi)
    dataLik = (N * log(mdl.beta) - mdl.beta * e_sq) / 2

    Phi_Sigma_Phit::Matrix = Phi_sub * Sigma_Phit
    PhiAllt_Phi_Sigma_Phit_Target::Vector = Phi_t * (Phi_Sigma_Phit * t)
    Phi_Sigma_Phit_PhiAll::Matrix = Phi_Sigma_Phit * Phi
    
    beta_sq = mdl.beta * mdl.beta
    Qs = zeros(M_all)
    Ss = zeros(M_all)
    
    for idx = 1:M_all
        phi_sq = Phi_t_Phi_diag[idx]
        phi_t = Phi_t_Target[idx]
        Q::Float64 = mdl.beta * phi_t - beta_sq * PhiAllt_Phi_Sigma_Phit_Target[idx]
        #S::Float64 = beta * phi_sq - beta_sq * sum(phi .* Phi_Sigma_Phit_PhiAll[:,idx])
        
        S::Float64 = dot(Phi[:,idx], Phi_Sigma_Phit_PhiAll[:, idx])
        #S::Float64 = 0
        #for innerIdx = 1:N
        #    term1::Float64 = Phi[innerIdx, idx]
        #    term2::Float64 = Phi_Sigma_Phit_PhiAll[innerIdx, idx]
        #    S += term1 * term2
        #end
        S = mdl.beta * phi_sq - beta_sq * S

        Qs[idx] = Q
        Ss[idx] = S
    end

    mdl.w[:] = 0
    mdl.w[indices] = Mu
    
    return Mu, U, Ui, alpha_sub, e, dataLik, Ss, Qs
end

function calculateW_MAP(Phi_sub::Matrix, t::Vector, alpha_sub::Vector, initVal::Vector)
    f(params) = rvmBernoulliPosterior(params, Phi_sub, t, alpha_sub)
    g!(params, storage) = rvmBernoulliPosterior_grad!(params, storage, Phi_sub, t, alpha_sub)
    h!(params, storage) = rvmBernoulliPosterior_hess!(params, storage, Phi_sub, t, alpha_sub)
    #@time opt = optimize(f, randn(length(alpha_sub)), method = :cg)
    #opt = optimize(f, g!, randn(length(alpha_sub)), method = :l_bfgs)
    opt = optimize(f, g!, h!, initVal, method = :newton)

    return opt.minimum
end

function calculateRVMPosterior!(mdl::RVMBernoulliModel, Phi::Matrix, t::Vector, Phi_t::Matrix, Phi_t_Target::Vector, Phi_t_Phi_diag::Vector)
    indices, Phi_sub, alpha_sub, A = getSubset(mdl.alpha, Phi)

    Mu = calculateW_MAP(Phi_sub, t, alpha_sub, mdl.w[indices])

    N, M = size(Phi_sub)
    
    mdl.w[:] = 0
    mdl.w[indices] = Mu
    y = sigmoid(Phi_sub * Mu)
    beta = y .* (1 - y)

    dataLik = dot(t, log(y)) + dot(1-t, log(1-y))

    betaBASIS_PHI = Phi_t * (Phi_sub .* (beta * ones(1, M)))
    
    e = t - y
    Qs = Phi_t * e
    #Ss = (Phi_t .^ 2) * beta - sum((betaBASIS_PHI * Ui) .^ 2, 2)
    term1::Vector = (Phi_t .* Phi_t) * beta
    #M_all = size(Phi, 2)
    #term1 = zeros(M_all)

    #for m = 1:M_all
    #    term1[m] = dot(Phi[:,m] .^ 2, beta)
    #end

    Phi_B = Phi_sub .* (beta * ones(1, M))
    Hessian = (transpose(Phi_B) * Phi_sub + A)
    U = chol(Hessian)
    Ui = inv(U)
    
    betaBASIS_PHI_Ui = betaBASIS_PHI * Ui
    term2::Vector = sum(betaBASIS_PHI_Ui .* betaBASIS_PHI_Ui, 2)[:,1]
    Ss = term1 - term2

    mdl.beta = beta
    
    return Mu, U, Ui, alpha_sub, e, dataLik, Ss, Qs
end

function calculateFactorQuantities(alpha::Float64, S::Float64, Q::Float64)
    s::Float64 = S
    q::Float64 = Q
    
    if !isinf(alpha)
        multTerm::Float64 = alpha / (alpha - S)
        s = multTerm * S
        q = multTerm * Q
    end

    factor::Float64 = q * q - s

    return s, q, factor
end     

function calculateRVMFullStatistics!(mdl::AbstractRVMModel, Phi::Matrix, t::Vector, Phi_t::Matrix, Phi_t_Target::Vector, Phi_t_Phi_diag::Vector)
    N, M = size(Phi)

    Mu, U, Ui, alpha_sub, e, dataLik, Ss, Qs  = calculateRVMPosterior!(mdl, Phi, t, Phi_t, Phi_t_Target, Phi_t_Phi_diag)
    
    # calculate log marginal likelihood
    logdetHOver2 = sum(log(diag(U)))
    logML::Float64 = dataLik - (transpose(Mu .* Mu) * alpha_sub)[1] / 2 + sum(log(alpha_sub))/2 - logdetHOver2
    diagC = sum(Ui .* Ui, 2)
    Gamma = 1 - alpha_sub .* diagC
    
    # calculate the statistics for each basis vector
    qs = zeros(M)
    ss = zeros(M)
    factors = zeros(M)
    
    for idx = 1:length(mdl.alpha)
        ss[idx], qs[idx], factors[idx] = calculateFactorQuantities(mdl.alpha[idx], Ss[idx], Qs[idx])
    end

    return Ss, Qs, ss, qs, factors, logML, e, Gamma
end

function calculateRVMAction(alpha::Float64, factor::Float64, isRequired::Bool)
    action::RVMAction = Hold

    used = !isinf(alpha)
    positiveFactor = factor > 1e-12
    if used && positiveFactor
        action = Reestimate
    elseif used && !positiveFactor && !isRequired
        action = Delete
    elseif !used && positiveFactor
        action = Add
    end
    
    return action
end

function calculateRVMDeltaML(action::RVMAction, alpha::Float64, Q::Float64, S::Float64, q::Float64, s::Float64, factor::Float64)
    DeltaML::Float64 = 0.0
    if action == Reestimate
        NewAlpha::Float64 = (s * s) / factor
        Delta::Float64 = (1 / NewAlpha - 1 / alpha)
        DeltaML = (Delta * (Q * Q) / (Delta * S + 1) - log(1 + S * Delta)) / 2
        #DeltaML = 0.5 * (s * Delta + log(1 + s / alpha) - log(1 + s / NewAlpha))
    elseif action == Delete
        DeltaML = -0.5 * (q * q / (s + alpha) - log(1 + s / alpha))
        #@printf "delete delta %g\n" DeltaML
    elseif action == Add
        quot::Float64 = Q * Q / S
        DeltaML = (quot - 1 - log(quot)) / 2
    end

    return DeltaML
end

function processRVMAction(action::RVMAction, alpha::Float64, s::Float64, factor::Float64)
    if action == Hold
        return alpha
    elseif action == Delete
        return Inf
    end

    return (s * s) / factor
end

function preprocessBasis(K::Matrix)
    N, M = size(K)

    Scales::Vector = transpose(sqrt(sum(K .* K, 1)))[:, 1]
    Scales[Scales .== 0] = 1;

    Ktf = deepcopy(K)
    
    for m in 1:M
        Ktf[:,m] = Ktf[:,m] / Scales[m]
    end
    return Ktf, Scales
end

function initTrain!(mdl::AbstractRVMModel, X::Matrix, t::Vector, options::AbstractRVMTrainingOptions)
    mdl.X = X
    Phi::Matrix = mdl.kernelFun(X, X)
    Phi, PhiScales = preprocessBasis(Phi)
    requiredIdxs = BitArray{1}(size(Phi, 2))
    requiredIdxs[:] = false

    return Phi, PhiScales, requiredIdxs
end

function initModel!(mdl::RVMGaussianModel, Phi::Matrix, t::Vector, options::RVMGaussianTrainingOptions)
    N, M = size(Phi)

    mdl.beta = 100 / var(t)
    mdl.alpha = Inf * ones(M)
    mdl.w = zeros(M)

    Phi_t_Target = transpose(Phi) * t
    alphaIdx = indmax(abs(Phi_t_Target))
    phi = Phi[:,alphaIdx]
    phi_sq::Float64 = sum(transpose(phi) * phi)
    phi_t::Float64 = sum(transpose(phi) * t)
    mdl.alpha[alphaIdx] = (phi_sq * mdl.beta) ^ 2 / ((phi_t * mdl.beta) ^ 2 - phi_sq * mdl.beta);    
end

function initModel!(mdl::RVMBernoulliModel, Phi::Matrix, t::Vector, options::RVMBernoulliTrainingOptions)
    N, M = size(Phi)
    
    mdl.alpha = Inf * ones(M)
    mdl.w = zeros(M)

    t_pseudoLinear = 2 * t - 1
    Phi_t_Target = transpose(Phi) * t_pseudoLinear
    alphaIdx = indmax(abs(Phi_t_Target))

    Phi_sub = Phi[:, alphaIdx]
    LogOut = (t_pseudoLinear * 0.9 + 1) / 2
    w = (Phi_sub \ (log(LogOut ./(1-LogOut))))[1]
    if w == 0
        w = 1
    end
    mdl.w[alphaIdx] = w
    mdl.alpha[alphaIdx] = 1 ./ (w .^ 2)

    if mdl.alpha[alphaIdx] < 1e-3
        mdl.alpha[alphaIdx] = 1e-3
    end

    if mdl.alpha[alphaIdx] > 1e3
        mdl.alpha[alphaIdx] = 1e3
    end
end

function updateSpecificModel!(mdl::RVMGaussianModel, options::RVMGaussianTrainingOptions, Phi::Matrix, t::Vector, Phi_t::Matrix, Phi_t_Target::Vector, Phi_t_Phi_diag::Vector, itIdx::Int64)
    if itIdx <= options.betaUpdateStart || rem(itIdx, options.betaUpdateFreq) == 0
        N = size(Phi, 1)
        Ss, Qs, ss, qs, factors, logML, e, Gamma = calculateRVMFullStatistics!(mdl, Phi, t, Phi_t, Phi_t_Target, Phi_t_Phi_diag)
        e_sq = sum(e .* e)
        mdl.beta = (N - sum(Gamma)) / e_sq
    end
end

function updateSpecificModel!(mdl::RVMBernoulliModel, options::RVMBernoulliTrainingOptions, Phi::Matrix, t::Vector, Phi_t::Matrix, Phi_t_Target::Vector, Phi_t_Phi_diag::Vector, itIdx::Int64)
    
end

function trainRVM!(mdl::AbstractRVMModel, X::Matrix, t::Vector, options::AbstractRVMTrainingOptions)
    Phi, PhiScales, requiredIdxs = initTrain!(mdl, X, t, options)
    #Phi = [ones(N) Phi];
    
    N, M = size(Phi)
    Phi_t = transpose(Phi)
    initModel!(mdl, Phi, t, options)
    #requiredIdxs[1] = true

    Phi_t_Target = transpose(Phi) * t
    Phi_t_Phi_diag = diag(transpose(Phi) * Phi)
    converged = false;
    
    DeltaML::Vector = zeros(M)
    actions = Array{RVMAction}(M)
    itIdx = 1

    Ss, Qs, ss, qs, factors, logML, e, Gamma = calculateRVMFullStatistics!(mdl, Phi, t, Phi_t, Phi_t_Target, Phi_t_Phi_diag)
    oldLogML = logML
    
    while !converged

        # process each basis
        for idx in 1:M
            actions[idx] = calculateRVMAction(mdl.alpha[idx], factors[idx], requiredIdxs[idx])
            DeltaML[idx] = calculateRVMDeltaML(actions[idx], mdl.alpha[idx], Qs[idx], Ss[idx], qs[idx], ss[idx], factors[idx])
            #@printf "%g " DeltaML[idx]
        end

        anyToAdd = any(actions .== Add)
        anyToDelete = any(actions .== Delete)

        if (anyToAdd && options.priorityAddition) || (anyToDelete && options.priorityDeletion)
            DeltaML[actions .== Reestimate] = 0
        end

        if (anyToAdd && options.priorityAddition && !options.priorityDeletion)
            DeltaML[actions .== Delete] = 0
        end

        if (anyToDelete && options.priorityDeletion && !options.priorityAddition)
            DeltaML[actions .== Add] = 0
        end
        
        # find the action that increases the log marginal likelihood the most
        basisIndex = indmax(DeltaML)
        oldAlpha = mdl.alpha[basisIndex]
        mdl.alpha[basisIndex] = processRVMAction(actions[basisIndex], mdl.alpha[basisIndex], ss[basisIndex], factors[basisIndex])
        #@printf "Action %s index %d alpha %g delta %g s %g factor %g\n" actions[basisIndex] basisIndex mdl.alpha[basisIndex] DeltaML[basisIndex] ss[basisIndex] factors[basisIndex]
        
        # model-specific update
        updateSpecificModel!(mdl, options, Phi, t, Phi_t, Phi_t_Target, Phi_t_Phi_diag, itIdx)
        Ss, Qs, ss, qs, factors, logML, e, Gamma = calculateRVMFullStatistics!(mdl, Phi, t, Phi_t, Phi_t_Target, Phi_t_Phi_diag)
        
        # check for convergence
        #if logML < oldLogML || (actions[basisIndex] == Reestimate && abs(log(oldAlpha) - log(mdl.alpha[basisIndex])) < options.logAlphaTol)
        if (actions[basisIndex] == Reestimate && abs(log(oldAlpha) - log(mdl.alpha[basisIndex])) < options.logAlphaTol)
            converged = true
        end
        #@printf "%d %g %d\n" itIdx logML/N sum(!isinf(mdl.alpha))
        oldLogML = logML
        itIdx += 1
    end

    mdl.w = mdl.w ./ PhiScales
    mdl.alpha = mdl.alpha ./ (PhiScales .* PhiScales)
end

function fit!(mdl::RVMGaussianModel, X::Matrix, t::Vector, options::RVMGaussianTrainingOptions = RVMGaussianTrainingOptions())
    trainRVM!(mdl, X, t, options)
end

function fit!(mdl::RVMBernoulliModel, X::Matrix, t::Vector, options::RVMBernoulliTrainingOptions = RVMBernoulliTrainingOptions())
    trainRVM!(mdl, X, t, options)
end

function predict(mdl::RVMGaussianModel, Xnew::Matrix)
    K_new = mdl.kernelFun(Xnew, mdl.X)

    y = K_new * mdl.w
end

function predict(mdl::RVMBernoulliModel, Xnew::Matrix)
    K_new = mdl.kernelFun(Xnew, mdl.X)

    y = sigmoid(K_new * mdl.w)
end
