
using Printf

SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- (λ/ρ),0) #Good one

#SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- ρ*λ,0)






function FFTOp(in,adj;normalize=true)
	norm = normalize ? sqrt(length(in[:])) : 1.
	if (adj)
		out = fft(in)/norm
	else
		out = bfft(in)/norm
	end
	return out
end	



function Sampling(in::Vector)

    cutoff = 1e-10
    i = 1

    wd = zeros(Real,length(in))
    for i = 1 : length(in)
        a =(in[i])^2;
        if (abs(a) > cutoff)
            wd[i] = 1.0;
        end
    end
    return wd;
end



function WeightingOp(in,adj;w=1.0)

        return in.*w;
end



function InnerProduct(in1,in2)
    
    return convert(Float32,real(sum(conj(in1[:]).*in2[:])))

end


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

function ConjugateGradients(b,operators,parameters;x0=0.0,Ni=10,μ=0.5,tol=1.0e-18)


    if x0 ≠ 0.0
        x0 = x0
    else
        x0=zeros(size(b));
    end



    J = Float64[]
    r= b - LinearOperator(x0,operators,parameters,adj=false); #Initial residual.   
    g = LinearOperator(r,operators,parameters,adj=true) -μ*x0; # ∇J= A'(Ax-y) ==> gradient
    m = zero(g) #New model.
    s = copy(g) #Copu  the gradient
    gamma = InnerProduct(g,g) # || ∇J||₂
    gamma00 = gamma;
    J0 = InnerProduct(r,r) # Initalize cost function.
    push!(J,1.0) # push value 1.0 into J
    k=0;




    @printf "==============================================\n";
    @printf "================= CGLS- Fernanda =======================\n";
    @printf "==============================================\n";

    @printf ("k              |grad|             |grad|/|grad_0|     J  \n"); 

    @printf(" %3.0f        %12.5g    %8.3g   %8.3g   \n", k,sqrt(gamma00),1, J0)
  
    Ni=5
    while k < Ni

        k=k+1

        t = LinearOperator(s,operators,parameters,adj=false); #Transform the gradient.
	    delta = InnerProduct(t,t) + μ*InnerProduct(s,s) #Compute delta.
	    #if delta <= tol
#	    println("delta reached tolerance, ending at iteration ",iter)
	    #    break;
	    #end
	    
        alpha = gamma/delta #Update alpha
 	    m = m + alpha*s #Update model with alfa and the gradient
	    r = r - alpha*t #Update the residual with alfa and projection of the gradient in the data
    	g = LinearOperator(r,operators,parameters,adj=true) #Compute a new gradient.
	    g = g - μ*m # 
	    
        gamma0 = copy(gamma) # ||∇J||
	    gamma = InnerProduct(g,g) #Compute || ∇Jk||₂
        Jk = InnerProduct(r,r) + μ*InnerProduct(m,m) #Update value of J for the iteration.
        push!(J,Jk/J0) #Push Jk into the vector J and normalize by J0
	    
        beta = gamma/gamma0 #Compue beta for the iteration.
	    s = beta*s + g #Update the gradient and goto t variable.

        norms0=sqrt(gamma00); norms=sqrt(gamma);
        @printf(" %3.0f          %12.5g      %8.3g   %8.3g \n", k,norms0,norms/norms0,Jk)

	
        #Stopping criteria: If the norm of the gradient is small enough, end.
        
       # if sqrt(gamma) <= sqrt(gamma00)*tol
        #    println("---------------------------------------------------")
          #  println("CG Iterations:")
         #   println("At iteration k=$k is ||∇Jₖ||₂² ≤ $tol* ||∇J₀||₂²")
          #  println("Inner loop ended.")
           # println("---------------------------------------------------")

	       # break;
        #end
        
    end

    return m, J
end


function CGLS(d_obs, operators,parameters; μ=0.5, Ni=100, tol=1.0e-15)

    #J=Float64[];
    m=zeros(size(d_obs));
    r= d_obs - LinearOperator(m,operators,parameters,adj=false);
    s =  LinearOperator(r,operators,parameters,adj=true) - μ*m;
    p=copy(s);

    gamma= InnerProduct(s,s);
    norms0=sqrt(gamma); #norm of the gradient is used to stop.
    norms=norms0;
    k=0;
    flag=0;
    J=zeros(Ni);
    J[1]=norm(r,2).^2

    @printf "==============================================\n";
    @printf "================= CGLS- joaquin =======================\n";
    @printf "==============================================\n";

    @printf ("k         ||∇J0||²       ||∇Jk||²    ||∇Jk||²/||∇J0||²      J   \n"); 

    @printf(" %3.0f  %12.5g   %12.5g    %8.3g    %8.3g \n", k,norms0,norms,norms/norms0,J[1])
  

    while k < Ni 


       
        k = k+1;

        q = LinearOperator(p,operators,parameters,adj=false);
        delta= InnerProduct(q,q) + μ*InnerProduct(p,p);

        if delta == 0
            delta=1.e-10;
        end

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
        #    println("Loop ended causde tolerance was reached",k)
         #   break;
        #end
        error = LinearOperator(m,operators,parameters,adj=false) - d_obs ;
        J[k] = sum( abs.(error[:]).^2 ) + μ*sum( (abs.(m)).^2);
       # J[k]=J[k]/J[1];
       # println(J[k])

       
       @printf(" %3.0f      %12.5g       %12.5g    %8.3g    %8.3g  \n", k,norms0,norms, norms/norms0,J[k])





    end

    return m, J
end


function ADMM_CGLS(m0,d_obs,operators,parameters; ρ= 1.0, μ= 1.8,tol=1e-8, Ni=5,Ne=50)
    
    Ji=Float64[]; Je=zeros(Ne);
    Je[1]= norm(d_obs,2).^2;


    u=zeros(size(m0)); 
    z=copy(m0);
    w=zero(u);


    for k=1:Ne;  
        b=  -1*LinearOperator(z.- u ,operators, parameters, adj=false) .+ d_obs;
        #w, Ji=ConjugateGradients(b,operators,parameters,Ni=Ni,μ=ρ,tol=1.0e-15)
        w, Ji= CGLS(b,operators, parameters; μ= ρ, Ni=Ni, tol=1.0e-10)
        x= w .+z .-u;
        z= SoftThresholding.( x .+ u,ρ,μ)  # z-update
        u= u .+ (x .-z); #dual update, lagrange multiploier

        aux=  LinearOperator(x,operators,parameters,adj=false)
        Je[k] = sum(abs.(aux .- d_obs).^2) + μ*sum(abs.(z))
        @printf("ADDM iteration   %3.0f ", k)
    end
     
    return z,Ji,Je

end




function MatrixMultiplyOp(in,adj;matrix=1)

	if (adj)
		out = matrix'*in
	else
		out = matrix*in
	end

	return out
end
