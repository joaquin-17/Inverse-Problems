

SoftThresholding(x,ρ,λ) = sign(x)*max(abs(x)- (ρ/(2*λ)),0)



function ADMM( m0,d_obs,operators,parameters; ρ= 1.0, μ= 1.8, Ni=1,Ne=50, tolerance=1.0e-5,)
   
   
    ρ=ρ;
    u=zeros(size(m0)); 
    z=copy(u);
    w=zero(u);
    #Initialize cost function with x0 misfit.
    x0=randn(size(d_obs)); J=zeros(Float64, Ne); #J[1]=norm(x0[:],2)^2; #
    Ji=zeros(Ne); 
    #Ji[1]=J[1]

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
        J[k] = sum(abs.(aux .- d_obs).^2) + μ*sum(abs.(z))

        #Tolerance check: Mofidy
        if k > 1;
            ΔJ = abs(J[k] - J[k-1]);
            if ΔJ < tolerance;
                  println(" ΔJ = $ΔJ  is  < $tolerance. Tolerance error reached adn loop ended with $k iterations.");
                  break
            end
        end
  
        @show k
    end
     
    return z, J, Ji

end



function ADMM2( m0,d_obs,operators,parameters; ρ= 1.0, μ= 1.8, Ni=1,Ne=50, tolerance=1.0e-5,)
   
   
    ρ=ρ;
    u=zeros(size(m0)); 
    z=copy(u);
    w=zero(u);
    #Initialize cost function with x0 misfit.
    #x0=randn(size(d_obs)); 
    J=zeros(Float64, Ne); #J[1]=norm(x0[:],2)^2; #
    Ji=zeros(Ne);
   # Ji[1]=J[1]

    k=0;
    
    while k < Ne
        
        k=k+1; #update counter

        b=  -1*LinearOperator(z.- u ,operators, parameters, adj=false) .+ d_obs; # this is the problem
        #d_obs= LinearOperator(z,operators,parameters, adj=false) 
        w,Ji=CG(b,operators,parameters;Niter=Ni,mu=ρ,tol=1.0e-15)
        #CGLS(m0,b,operators, parameters; μ= ρ, Ni=Ni, tol=1.0e-15)
        x= w .+z .-u;
        z= SoftThresholding.( x .+ u,ρ,μ)  # z-update
        u= u .+ (x .-z); #dual update, lagrange multiploier
        
        aux=  LinearOperator(z,operators,parameters,adj=false)
        residual= abs.(aux .- d_obs);

        J[k] = sum(residual.^2) + μ*sum(abs.(z))

        #Tolerance check: Mofidy
        if k > 1;
            ΔJ = abs(J[k] - J[k-1]);
            if ΔJ < tolerance;
                  println(" ΔJ = $ΔJ  is  < $tolerance. Tolerance error reached adn loop ended with $k iterations.");
                  break
            end
        end
  
        @show k
    end
     
    return z, J, Ji

end



function ADMMuntouched( m0,d_obs,operators,parameters; ρ= 1.0, μ= 1.8, Ni=1,Ne=50, tolerance=1.0e-5,)
   
   
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

        #Tolerance check: Mofidy
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
