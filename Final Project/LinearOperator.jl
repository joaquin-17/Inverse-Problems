"""
LinearOperator(in,operators,parameters,adj=true)

This function applies foward or adjoint linear operators on the flight to the input `in`.

Returns as output the resultant array.


# Arguments
- `in::AbstractArray{T,N}`: input seismic data as ND array
- `operators::Vector{Functions}`: A vector containing the operators to apply on the flight.
- `parameters::Vector{Dict{Symbol, V} where V}`: A dictionary that contains the parameters for the operators.
- `adj::Bool` : A flag to apply the foward (false) or the adjoint operator (true). The adjoint operator is chosen by default.
"""

function LinearOperator(in,operators,parameters;adj=true)
    if adj == true
        d=copy(in)
        m =[];
        for j=1:1:length(operators)
            op = operators[j]
            m = op(d,true;parameters[j]...)
            d = copy(m)
        end
    return m
    else
        m = copy(in)
        d = [];
        for j=length(operators):-1:1
            op=operators[j]
            d=op(m,false;parameters[j]...)
            m=copy(d);
        end
        return d
    end
end