



#Square of the norm of a vector
#L2(in::Vector)= sum.((in).^2);


#loss functions
L2Loss(in::Vector)= (1/2)*(in).^2;
L1Loss(in::Vector) = abs.(in);
CauchyLoss(in::Vector; δ::Real=1.0) = log.( 1 .+ (in/δ).^2)
HuberLoss(in::Vector; δ::Real=1.0) = [abs(k) ≤ δ ? (1/2)*(k)^2 : δ*(abs(k)-(1/2)*δ)  for k in in]

#influence
L2influence(in::Vector)= in;
L1influence(in::Vector) = (abs.(in)./in);
Cauchyinfluence(in::Vector; δ::Real=1.0) = ( in ./ (δ^2 .+(in).^2))
Huberinfluence(in::Vector; δ::Real=1.0) = [ abs(k) ≤ δ ? k : δ*(k/(abs(k))) for k in in];




L2weights(in::Vector; δ::Real=1.0)= ones(length(in));# L2 norm for IRLS, is the identity
L1weights(in::Vector;δ::Real=1.0) = sum.(abs.(in));
Cauchyweights(in::Vector; δ::Real=1.0) = sum.( δ^2 .+ (in).^2);
Huberweights(in::Vector; δ::Real=1.0) = [abs(k) ≤ δ ? 1 : abs(k)/δ  for k in in]
Weights(in::Vector; norma::Function, ϵ::Real, δ::Real=1.0) =  1 ./( norma(in,δ=δ).+ ϵ)

function IRLS(d::Vector,A::Matrix,m0::Vector; norma::Function=L2, ϵ::Real=1e-4, δ::Real=1.0,Ni::Int=10)
   
   
   models=zeros(Float64,(length(m0),Ni+1))
   models[:,1] = m0;
   
   for i=1:Ni
      d_pred=A*models[:,i]; #predicted data
      error= d .- d_pred; #error vector
      W=diagm(Weights(error,norma=norma,ϵ=ϵ,δ=δ)) ; #weights
      models[:,i+1] = inv(A'*W*A)*(A'*W)*d; #Inversio
   end
   return models

end


