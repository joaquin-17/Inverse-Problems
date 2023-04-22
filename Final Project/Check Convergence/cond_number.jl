function k(G::Matrix, ρ::Float64)

    I=diagm(ones(size(G,1)));
    E=eigen(G+ ρ*I);
    λ=E.values;
    κ= (maximum(λ)+ ρ )/ (minimum(λ) + ρ)

    return λ, κ

end