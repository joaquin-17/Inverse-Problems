
function IRLS(;d::Vector,A::Matrix,m0::Vector, norma::Function, ϵ::Real=1e-4, Ni::Int=100)
   
   
#models=zeros(Float64,(length(m0),Ni+1))
#models[:,1] = m0;
m=m0
for i=1:Ni
   d_pred=A*m; #predicted data
   error= d .- d_pred; #error vector
   W=diagm(sqrt.(1 ./ norma(error) .+ ϵ)); #weights
   m = inv(A'*W*A)*(A'*W)*d; #Inversion
end

return m#models

end



#Square of the norm of a vector
L2(in::Vector)= sum.((in).^2);

L1(in::Vector)= sum.(abs.(in));