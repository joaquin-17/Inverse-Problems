Weights(in::Vector; norma::Function, ϵ::Real, δ::Real=1.0) =  1 ./( norma(in,δ=δ).+ ϵ)

function IRLSCG(t::Vector,D::Matrix,m0::Vector; ϵ::Real=1e-9, δ::Real=1.0,Ne::Int=10)

    m=m0;
    slo=reshape(m,(gz,gx))
    (Dx,Dz)=Derivatives(slo,1);  
    β=6.8;
    Dxm0=β*(Dx*m);
    Dzm0=β*(Dz*m);
    
    for i=1:Ne
        h=Dx*m; v=Dz*m;
        Wx=diagm(1 ./(δ^2 .+ h.^2 .+ ϵ));
        Wz=diagm(1 ./(δ^2 .+ h.^2 .+ ϵ));
        A=vcat(D,β*Wx*Dx,β*Wz*Dz); #Concatenation of forward models. Left side of the equation.
        y=vcat(t,Dxm0,Dzm0); #right side of the equation
        m=cgaq(A,y,nt)
    end

    return m
end











#=
"""
    IRLS(d,operators,parameters;<keyword arguments>)
Non-quadratic regularization with Iteratively Reweighted Least Squares (IRLS).
# Arguments
- `Niter_external=3`
- 'Niter_internal=10'
- `mu=0`
"""
function IRLS(d,operators,parameters;Niter_external=3,Niter_internal=10,mu=0)

	cost = Float64[]
	weights = ones(Float64,size(d))
	parameters[end][:w] = weights
	m = []
	for iter_external = 1 : Niter_external
		v,cost1 = ConjugateGradients(d,operators,parameters,Niter=Niter_internal,mu=mu)
		append!(cost,cost1)
		m = v .* weights
		weights = abs.(m./maximum(abs.(m[:])))
		parameters[end][:w] = weights
	end

	return m, cost

end




"""
    ConjugateGradients(d,operators,parameters;<keyword arguments>)
Conjugate Gradients following Algorithm 2 from Scales, 1987.
The user provides an array of linear operators. Verify that linear operator(s) pass the dot product.
See also: [`DotTest`](@ref)
# Arguments
- `Niter=10` : Number of iterations
- `mu=0`
- `tol=1.0e-15`
"""
function ConjugateGradients(d,operators,parameters;Niter=10,mu=0,tol=1.0e-15)

    cost = Float64[]
    r = copy(d)
    g = LinearOperator(r,operators,parameters,adj=true)
    m = zero(g)
    s = copy(g)
    gamma = InnerProduct(g,g)
    gamma00 = gamma
    cost0 = InnerProduct(r,r)
    push!(cost,1.0)
    for iter = 1 : Niter
	t = LinearOperator(s,operators,parameters,adj=false)
	delta = InnerProduct(t,t) + mu*InnerProduct(s,s)
	if delta <= tol
#	    println("delta reached tolerance, ending at iteration ",iter)
	    break;
	end
	alpha = gamma/delta
	m = m + alpha*s
	r = r - alpha*t
	g = LinearOperator(r,operators,parameters,adj=true)
	g = g - mu*m
	gamma0 = copy(gamma)
	gamma = InnerProduct(g,g)
        cost1 = InnerProduct(r,r) + mu*InnerProduct(m,m)
        push!(cost,cost1/cost0)
	beta = gamma/gamma0
	s = beta*s + g
	if (sqrt(gamma) <= sqrt(gamma00) * tol)
	    println("tolerance reached, ending at iteration ",iter)
	    break;
	end
    end

    return m, cost
end

=#