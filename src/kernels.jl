function kernelLinear(X1::Matrix, X2::Matrix)
    N1 = size(X1, 1)
    N2 = size(X2, 1)

    S = zeros(N1, N2)

    for idx1 in 1:N1
        X1Vec::Vector = transpose(X1[idx1,:])[:,1]
        for idx2 in 1:N2
            X2Vec::Vector = transpose(X2[idx2,:])[:,1]
            S[idx1, idx2] = dot(X1Vec, X2Vec)
        end
    end

    return S
end

function kernelGaussian(X1::Matrix, X2::Matrix, sigma::Float64)
    N1 = size(X1, 1);
    N2 = size(X2, 1);

    S = zeros(N1, N2);

    for idx1 in 1:N1
        X1Vec = X1[idx1,:]
        for idx2 in 1:N2
            diff::Vector = transpose(X1Vec - X2[idx2,:])[:,1]
            S[idx1, idx2] = dot(diff, diff);
        end
    end

    K = exp(-1/(2*sigma^2) * S);
end
