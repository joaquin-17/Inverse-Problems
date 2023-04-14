
function ErrorRMS(d_ideal, d_rec; relative= false)

           N=size(d_ideal)
           e = d_rec .- d_ideal;


end

         
           


function SoftThresholding(u,η,λ)
    
    S=sign(u)*max(abs(u)- η*λ,0)
    
    return S;

end


    
function PowerMethod(x0,operators,parameters)
    
    x= x0;
    α=0.0;
    for k = 1:10;
        aux=LinearOperator(x,operators,parameters,adj=false)
        y=LinearOperator(aux,operators,parameters,adj=true)
        n = norm(y,2);
        x = y/n;
        α = n;
    end
    return α
end



function ISTA(x0,y,operators,parameters; λ= 0.5,Ni=150,tolerance=1.0e-5)

    x0=randn(size(x0));
    α = PowerMethod(x0,operators,parameters); #x0, operators, parameters
    η= 0.95/α;
    J=zeros(Float64, Ni);
    m = zeros(Float64,size(x0))

     
    k=0;
    
    while k < Ni  #&& err > tolerance
        
        k=k+1
        mk = copy(m);
        aux1= LinearOperator(mk,operators,parameters,adj=false) #A*x
        aux2= aux1 .-y; #(A*x.-y)
        aux3= LinearOperator(aux2,operators,parameters,adj=true) #A'*(A*x.-y)==> this is the gradient!;
        ∇fₖ= aux3; #change name
        u= mk  .-  η* ∇fₖ;
        m=SoftThresholding.(u,η,λ);
        aux4=  LinearOperator(m,operators,parameters,adj=false)
        aux5= (abs.(aux4 .- y)).^2;
        #aux6= abs.((aux5')*(aux5));
        J[k] = sum(aux5) + λ*sum(abs.(m))

        #Tolerance check:
        if k >1;
            ΔJ= abs(J[k] - J[k-1]);
            if ΔJ < tolerance;
                println(" ΔJ = $ΔJ  is  < $tolerance. Tolerance error reached adn loop ended with $k iterations.")
            break
            end
        end

        @show k

    
    end
    return m, J
end






function ISTA(x0,y,operators,parameters; μ= 0.5,Ni=150,tolerance=1.0e-5)

    x0=randn(size(x0));
    α = PowerMethod(x0,operators,parameters); #x0, operators, parameters
    η= 0.95/α;
    #Initialize cost function with x0 misfit.
    J=zeros(Float64, Ni); J[1]=norm(x0[:],2)^2; #
    m = zeros(Float64,size(x0))
    t=1.0;
    T=μ/(2*η);


     
    k=0;
    
    while k < Ni  #&& err > tolerance
        
        k=k+1
        mk = copy(m);
        aux1= LinearOperator(mk,operators,parameters,adj=false) #A*x
        aux2= aux1 .-y; #(A*x.-y)
        aux3= LinearOperator(aux2,operators,parameters,adj=true) #A'*(A*x.-y)==> this is the gradient!;
        ∇fₖ= aux3; #change name
        u= mk  .-  η* ∇fₖ;
        m=SoftThresholding.(u,η,μ);
        aux4=  LinearOperator(m,operators,parameters,adj=false)
        aux5= (abs.(aux4 .- y)).^2;
        #aux6= abs.((aux5')*(aux5));
        J[k] = sum(aux5) + μ*sum(abs.(m))

        #Tolerance check:
        if k >1;
            ΔJ= abs(J[k] - J[k-1]);
            if ΔJ < tolerance;
                println(" ΔJ = $ΔJ  is  < $tolerance. Tolerance error reached adn loop ended with $k iterations.")
            break
            end
        end

        @show k


        #Update point to improve convergence rate  (ISTA --> FISTA)
        
        #t_k = t;
        #t = (0.5)*(1.0 + sqrt(1.0 + 4.0*(t_k)^2))
        #yk = m + (t_k-1.0)/t*(m-mk); 
        
        

    
    end
    return m, J
end












function FISTA(x0,yobs,operators,parameters,μ,Ni,tolerance)
    
    
    x0 = randn(size(x0)); 
    α = 1.05*power_method(x0,operators,parameters);
    #Initialize:
    J=zeros(Float64, Ni +1);
    m=zeros(Float64,size(x0));
    yk=copy(m);


    misfit=norm(x0[:])^2
    J[1]= misfit;
    t=1.0;
    err= 1e4;
    T=μ/(2*α);
    
    #Iterative gradient with soft thresholding:
    
    k=0;
    
    while k < Ni  && err > tolerance
        
        k=k+1
        #Update model with FISTA
        mk = copy(m);
        adjoint =  LinearOperator(yk,operators,parameters,adj=false) 
        aux1= yobs .- adjoint
        forward= LinearOperator(aux1,operators,parameters,adj=true) #a
        m=copy(forward);  
        m=soft_thresholding(yk .+ (m/α) , T); 
        
        J[k] = sum(abs.(yobs .- adjoint).^2) + μ*sum(abs.(m)); # Cost function J= || yobs-  Operators(x) ||₂ + μ ||x||₁   
        

        if k> 1 ; err= abs(J[k] - J[k-1])/((J[k]+J[k-1])/2);

        end

        if err < tolerance
            println(" error = $err  is < $tolerance. Loop ended with $k iterations.")
            break
        end

        #Update point to improve convergence rate  (ISTA --> FISTA)
        
        tk = t;
        t = (0.5)*(1.0 + sqrt(1.0 + 4.0*tk^2))
        yk = m + (tk-1.0)/t*(m-mk); 
        
        
        @show k

    end


    return m, J
end



function soft_thresholding(in, T::Float64)
    
    tmp = abs.(in) .- T
    tmp = (tmp .+ abs.(tmp)) / 2
    out = sign.(in) .* tmp
    
    return out
end

function power_method(x0,operators,parameters)
    
    x= x0;
    α=0.0;
    
    for k = 1:10;
        
        aux=LinearOperator(x,operators,parameters,adj=false)
        y=LinearOperator(aux,operators,parameters,adj=true)
        n = norm(y,2);
        x = y/n;
        α = n;
    end

    return α


end