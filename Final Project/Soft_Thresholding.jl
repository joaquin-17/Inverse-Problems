function soft_thresholding(in, T::Float64)
    
    tmp = abs.(in) .- T
    tmp = (tmp .+ abs.(tmp)) / 2
    out = sign.(in) .* tmp
    
    return out
end