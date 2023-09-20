"""
    CalculateSampling(in)
Calculate the sampling operator of an n-dimension input. The output has the
same size as the input. 
"""
function CalculateSampling(in)

cutoff = 1e-10
	itrace = 1

	n=size(in)
	in=reshape(in,n[1],:)
        wd = zeros(Float32,size(in))
	n2=size(in,2)
	for itrace = 1 : n2
		a = sum(in[:,itrace].*in[:,itrace])
		if (a > cutoff)
			wd[:,itrace] .= 1.
		end
	end
	wd=reshape(wd,n)
	return wd;

end
function ErrorRMS(y, yp; relative= false)

           dev = yp .- y; #deviation
           e_rms= sqrt(sum(dev.^2)/prod(size(y))); #RMS deviation


           if relative == true # % RMS deviation
            num=e_rms; 
            den= sqrt(sum((yp).^2));
            p= num/den

            return p

           end


           return e_rms

end



function Error(yi, yp; relative= false)

    dev = yp .- yi; #deviation
    e=sqrt(sum( abs.((dev).^2))); 

    if relative == true # % RMS deviation
        num=e; 
        den= sqrt(sum((yi).^2));
        p= num/den
     return p

    end

    return e

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



function ISTA(x0,y,operators,parameters; λ= 0.5,Ni=50,tolerance=1.0e-3)

    x0=randn(size(x0));
    α = PowerMethod(x0,operators,parameters); #x0, operators, parameters
    η= 0.95/α;
    J=zeros(Float64, Ni);
    m = zeros(Float64,size(x0))
    t=1.0;
    ΔJ= 1e4;
    yk=copy(m);
    perc_error = 1.0
    #T=μ/(2*α);

     
    k=0;
    
    while k < Ni  #&& err > tolerance
        
        k=k+1


        mk = copy(m);
        aux1= LinearOperator(yk,operators,parameters,adj=false) #A*x
        aux2= aux1 .-y; #(A*x.-y)
        aux3= LinearOperator(aux2,operators,parameters,adj=true) #A'*(A*x.-y)==> this is the gradient!;
        ∇fₖ= aux3; #change name ==> until here, I am pretty sure is working.
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

        
        #Update point to improve convergence rate  (ISTA --> FISTA)
        
        tk = t;
        t = (0.5)*(1.0 + sqrt(1.0 + 4.0*tk^2))
        yk = m + (tk-1.0)/t*(m-mk); #where does yk goes?

        @show k

    
    end
    return m, J
end




function FISTA(x0,y,operators,parameters; λ= 0.5,Ni=100,tolerance=1.0e-3)



    println("")
    println(" ==================================================")
    println(" Fast Iterative Soft Thresholding Algorithm (FISTA)")
    println(" ==================================================")
    println("")

    #Initialize
    x0=randn(size(x0));
    α = PowerMethod(x0,operators,parameters);
    η= 0.95/α;
    J=zeros(Float64, Ni+1);
    J[1]=norm(x0[:])^2;
    #misfit0= J[1]; 
    ΔJ = J[1];
    m = zeros(Float64,size(x0));
    t=1.0;
    p = copy(m);
   
  


    k=0;
    while k < Ni

            #Update counter
            k=k+1       
            
            ##Update model with FISTA##
            m_old = m; #mk= m;
            Am=LinearOperator(p,operators,parameters,adj=false);
            r= Am .-y; #residual= (A*x.-y)
            ∇fₖ=LinearOperator(r,operators,parameters,adj=true); #A'(r)
            m = p .- η*∇fₖ; #update
            m = SoftThresholding.(m,η,λ); #soft-thershold update
            
            ##FISTA acceleration step
            t_old=t;
            t = (0.5)*(1.0 + sqrt(1.0 + 4.0*(t_old)^2));
            p = m +((t_old-1.0)/t)*(m-m_old);

            #Update cost function
            yp= LinearOperator(m,operators,parameters,adj=false); #predicted
            misfit_term= sum(abs.(yp .- y).^2); #|| Am - y ||₂²
            regularization_term= sum(abs.(m)); #|| m ||₁
            J[k+1] = (1/2)*misfit_term + λ*regularization_term; #loss function
            #misfit= J[k+1]
            #ΔJ= (abs(misfit0 - misfit))/misfit0; #Normalize 
           
            #FISTA tolerance criteria:
            
            if k> 1
                 ΔJ= abs(J[k] - J[k-1])/((J[k]+J[k-1])/2);
                 if ΔJ < tolerance
                    println("Loop ended at $k iterations.")
                    println("REASON: ")
                    println(" ΔJ = $ΔJ  is < than the tolerance = $tolerance used.")
                    break
                end
            end
            println("-----------------------------------------------")
            print("Monitoring iteration number "); @show k 
            print("Monitoring differences in cost function throughout the iterations "); @show ΔJ
            println("-----------------------------------------------")

    end
        return m, J
end