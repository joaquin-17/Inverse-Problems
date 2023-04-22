

SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- (λ/ρ),0)


#=

function FFTOp(in,adj;normalize=true)
	norm = normalize ? sqrt(length(in[:])) : 1.
	if (adj)
		out = fft(in)/norm
	else
		out = bfft(in)/norm
	end
	return out
end	



function WeightingOp(in,adj;w=1.0)

        return in.*w;
end

=#

function InnerProduct(in1,in2)
    
    return convert(Float32,real(sum(conj(in1[:]).*in2[:])))

end

#=

function LinearOperator(in,operators,parameters;adj=true)
	if adj
		d = copy(in)
		m = [];
		for j = 1 : 1 : length(operators)
			op = operators[j]
			m = op(d,true;parameters[j]...)
			d = copy(m)
		end
		return m
	else
		m = copy(in)
		d = [];
		for j = length(operators) : -1 : 1
			op = operators[j]
			d = op(m,false;parameters[j]...)
			m = copy(d)
		end
		return d
	end

end
=#




"""
%CGLS: Solves for the minimum of J = || A x - b ||_2^2  + mu ||x||_2^2 
%      via the method of conjugate gradients for least-squares problems. The 
%      matrix A is given via an  operator  and apply on the flight by
%      user-defined function "operator" with parameters "Param".

"""

function CGLS(m0,d_obs, operators,parameters; μ=0.5, Ni=100, tol=1.0e-15)

    m=m0
    #m=zeros(size(m0));
    r= d_obs - LinearOperator(m,operators,parameters,adj=false);
    s =  LinearOperator(r,operators,parameters,adj=true) - μ*m;
    p=copy(s);

    gamma= InnerProduct(s,s);
    norms0=sqrt(gamma); #norm of the gradient is used to stop.
    k=0;
    flag=0;
    J=zeros(Ni);
    J[1]=norm(m,2)


    
    while k < Ni && flag == 0
        
        q = LinearOperator(p,operators,parameters,adj=false);
        delta= InnerProduct(q,q) + μ*InnerProduct(p,p);

        ·#if delta <= tol
         #   #println("delta reached tolerance, ending at iteration ",iter)
         #   break;
        #end

        alpha= gamma/delta;
        m = m + alpha*p;
        r = r - alpha*q;
        s =  LinearOperator(r,operators,parameters,adj=true)  - μ*m;
        gamma1  = InnerProduct(s,s);
        norms  = sqrt(gamma1);
        beta = gamma1/gamma;
        gamma = gamma1;
        p = s + beta*p;
        #if norms <= norms0*tol
         #   println("Loop ended causde tolerance was reached",k)
         #   break;
        #end
        k = k+1;
        println(k)
        error = LinearOperator(m,operators,parameters,adj=false) - d_obs ;
        J[k] = sum( abs.(error[:]).^2 ) + μ*sum( (abs.(m)).^2);

    end

    return m, J
end


function ADMM( m0,d_obs,operators,parameters; ρ= 1.0, μ= 1.8, Ni=1,Ne=50, tolerance=1.0e-5,)
   
   
    ρ=ρ;
    u=zeros(size(m0)); 
    z=copy(u);
    w=zero(u);
    #Initialize cost function with x0 misfit.
    x0=randn(size(d_obs)); J=zeros(Float64, Ne); J[1]=norm(x0[:],2)^2; #
    Ji=zeros(Ne);
    Ji[1]=J[1]

    k=0;
    
    while k < Ne
        
        k=k+1; #update counter

        b=  -1*LinearOperator(z.- u ,operators, parameters, adj=false) .+ d_obs; # this is the problem
        #d_obs= LinearOperator(z,operators,parameters, adj=false) 
        w,Ji= CGLS(m0,b,operators, parameters; μ= ρ, Ni=Ni, tol=1.0e-15)
        x= w .+z .-u;
        z= SoftThresholding.( x .+ u,ρ,μ)  # z-update
        u= u .+ (x .-z); #dual update, lagrange multiploier
        
        aux=  LinearOperator(z,operators,parameters,adj=false)
        J[k] = sum(abs.(aux .- d_obs)).^2 + μ*sum(abs.(z))

        #Tolerance check:
        if k > 1;
            ΔJ = abs(J[k] - J[k-1]);
            if ΔJ < tolerance;
                  println(" ΔJ = $ΔJ  is  < $tolerance. Tolerance error reached adn loop ended with $k iterations.");
                  break
            end
        end
  
        @show k
    end
     
    return z, J,Ji

end
