

SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- (ρ/(2*λ)),0)



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

function CGLS(m0,d_obs, operators,parameters; μ=0.5, Ni=100, tol=1.0e-6)

    m=copy(m0);
    #m=zeros(size(m0));
    r= d_obs - LinearOperator(m,operators,parameters,adj=false); #residual
    s =  LinearOperator(r,operators,parameters,adj=true) - μ*m;  # ∇J
    p=copy(s);

    gamma= InnerProduct(s,s); #  ||∇J||²₂   
    norms0=sqrt(gamma);  #||∇J||₂ => norm of the gradient at the beginining.
    
    k=0; #Initialize counter
    #flag=0;
    J=zeros(Ni);
    J[1]=(norm(r,2))^2


    
    while k < Ni #&& flag == 0
        
        q = LinearOperator(p,operators,parameters,adj=false);
        delta= InnerProduct(q,q) + μ*InnerProduct(p,p);

        if delta <= tol
            #println("delta reached tolerance, ending at iteration ",iter)
            break;
        end

        alpha= gamma/delta; #  ||∇J||²₂  /  ||∇J||²₂   +μ 
        m = m + alpha*p;
        r = r - alpha*q;
        s =  LinearOperator(r,operators,parameters,adj=true)  - μ*m;
        gamma1  = InnerProduct(s,s); #update  ||∇J||²₂   
        norms  = sqrt(gamma1); #update ||∇J||₂   
        beta = gamma1/gamma; # |∇Jk+1||₂ /|∇Jk||₂ 
        gamma = gamma1; # save the new cost function
        p = s + beta*p; 
        if norms <= norms0*tol
            println("Loop ended at iteration number k=",k)
            println("At iteration k=$k is ||∇Jₖ||₂² ≤ $tol* ||∇J₀||₂²")
            break;
         end
        k = k+1;
        println(k)
        error = LinearOperator(m,operators,parameters,adj=false) - d_obs ;
        J[k] = sum( abs.(error[:]).^2 ) + μ*sum( (abs.(m)).^2);

    end

    return m, J
end


function ADMM( m0,d_obs,operators,parameters; ρ= 1.0, μ= 1.8, Ni=1,Ne=50, tolerance=1.0e-5,)
   
    #CHANGE μ to be λ but consider how that will affect the CGLS.
    #change z to be m 
    #Change m0 to be x0
   
    ρ=ρ;
    #Initialize variables
    u=zeros(size(m0));
    z=copy(u);
    w=zero(u); #Lagrange multiplier
    #Initialize cost function with x0 misfit.
    x0=randn(size(d_obs));  #SO FAR ONLY USED TO INITIALIZE THE COST FUNCTIONS.

    #External cost function for ADMM
    Je=zeros(Float64, Ne+1);
    Je[1]=norm(x0[:],2)^2;
    #Internal cost function for CGLS
    Ji=zeros(Ne);
    Ji[1]=norm(x0[:],2)^2;
    
    #Initialize difference 
    ΔJe = Je[1];
    #ΔJi = Ji[1];


    k=0;
    
    while k < Ne
        
        k=k+1; #update counter

        #Update model with ADMM
        b=  -1*LinearOperator(z.- u ,operators, parameters, adj=false) .+ d_obs; # Call CGLS,this is the problem, why?
        w, Ji= CGLS(m0,b,operators, parameters; μ= ρ, Ni=Ni, tol=1.0e-15)
        x= w .+z .-u; #Variable change
        z= SoftThresholding.( x .+ u,ρ,μ)  #Soft-Thersholding, z-update
        u= u .+ (x .-z); #Dual update ==> lagrange multiploier
        

        #Update external cost function
        yp=  LinearOperator(z,operators,parameters,adj=false); #predicted
        misfit_term= sum(abs.(yp .- d_obs).^2); #|| Am - y ||₂²
        regularization_term= sum(abs.(z)); #|| m ||₁
        Je[k+1] = (1/2)*misfit_term + μ*regularization_term; # external loss function

        #ADMM tolerance criteria
        if k> 1
            ΔJe= abs(Je[k] - Je[k-1])/((Je[k]+Je[k-1])/2);
            if ΔJe < tolerance
               println("Loop ended at $k iterations.")
               println("REASON: ")
               println(" ΔJe = $ΔJe  is < than the tolerance = $tolerance used.")
               break
           end
       end
  
       println("-----------------------------------------------")
       print("Monitoring iteration number "); @show k 
       print("Monitoring differences in the cost internal function throughout the iterations "); @show ΔJi
       print("Monitoring differences in the cost external function throughout the iterations "); @show ΔJe
       println("-----------------------------------------------")

    end
     
    return z, Je

end
