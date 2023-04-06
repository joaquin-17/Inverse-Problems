#Forward model
using SparseArrays


function DistanceMatrix(sources, receivers, grid)
    
    #Define grid:

    gx,gz=grid

    xgrid=LinRange(0.0,300,gx+1);
    dx=xgrid[2]-xgrid[1];
    xgrid=collect(xgrid[2:end-1]);
    zgrid=LinRange(0.0,400,gz+1);
    dz=zgrid[2]-zgrid[1];
    zgrid=collect(zgrid[2:end-1]);
        
    #Allocation variables
    D=zeros(Float64,(length(sources[:,1])*length(receivers[:,1]),gx*gz));
    ray_number= 0; 
    js=Float64[];
    is=Float64[];
    vs=Float64[]



    
    
     for s in 1:length(sources[:,1])
        for r in 1:length(receivers[:,1])
        
            point1=[receivers[r,2] sources[s,2]];
            point2=[receivers[r,1] sources[s,1]];
                
            #Compute m and b in a function get ray param
            m=(receivers[r,2]-sources[s,2])/(receivers[r,1]-sources[s,1]); #Slope of the ray
            b=receivers[r,2]-(receivers[r,1]*m); #Intercept of the ray
        
            #Compute grid points involved in calculations, Put into a matrix
            
            cx=((zgrid.-b)./m); 
            cz=((m.*xgrid).+b); 
               
            t1=[cz' zgrid' point1;  xgrid' cx' point2];
            CP=unique(sortslices(round.(t1;digits=2),dims=2),dims=2);

            indx1= findall(x->  x > 300.0 || x < 0.0 ,CP[2,:]) #rplace with variables
            indx2= findall(x->  x > 400.0 || x < 0.0 ,CP[1,:]) #rplace with variables
            
            indx=[indx1; indx2]

            aux=zeros(length(indx))
            
            for i in 1:length(indx)
                aux[i]=indx[i][1]; #loop to extract the indices of the collumns that are out of bounds
            end
                
            CP =CP[:, 1:end .∉[aux]]; #Final Cross matrix without undesired value
            
            #compute the distances
        
            dist=zeros(length(CP[1,:])-1);

            for i in 1:length(CP[1,:])-1
                dist[i]=sqrt(((CP[1,i+1]-CP[1,i])^2)+(CP[2,i+1]-CP[2,i])^2);
            end

            #Compute the centers;
        
            centers=zeros(2,length(CP[1,:])-1);

            for i in 1:2
                for j in 1:length(CP[1,:])-1
                    centers[i,j]=((CP[i,j+1]-CP[i,j])/2)+CP[i,j] #Calculation of segment center
                end
            end
        
            cell_id=zeros(length(dist))
     
            for i in 1:length(dist)
                cell_id[i]=ceil(centers[1,i]/dz)+(((ceil(centers[2,i]/dx))-1)*gz)
            end

    
            ray_number= ray_number+1;

            for j=1:length(cell_id)
                D[ray_number,Int(cell_id[j])] = dist[j]
            end

            for i in 1:length(cell_id)
                js=push!(js,cell_id[i]);
                is=push!(is,ray_number);
                vs=push!(vs,dist[i]);
            end 
        end
    
    end

    return D, is, js, vs
    
end



function DLS(D,t,μ,m0,gz,gx)


    Id=Matrix{Int64}(I,(gz*gx,gz*gx))
    A=inv(D'*D + μ*Id);
    g=(D'*t + μ*m0)
    m=A*g;

    m=reshape(m,(gz,gx));
    #dv= (1 ./ dm)

    #m= dm .+ reshape(m0,(gz,gx));
    v= (1 ./ m);

    return  m, v
    
end



function cgaq(A,y,it)
    #We have the output (y) and the forward operator A.
    #We will try to invert for x, which is known to be a vector of ones.
     #Initial guess of zeros, true answer are ones.
     x=zeros(length(A[1,:]));
     error=zeros(length(it))
     #it=length(x)
     s=y.-(A*x);              #s represents the residual error, this lives in the space of the data
     p=A'*s;                   #By applying A', we are moving the residual back to the model space. Getting a model error(?)
     r=p;                      #    ??
     q=A*p                     #We are taking the model error(?) and we are projecting it back to the data space.
     old=r'*r;                 #norm squared of the model error(?) are we looking to make this small as well?
    #Conjugate gradient loop
    for k in 1:it
         alpha=(r'*r)/(q'*q);   #Ratio between the sq norm of the model error(?) and the sqnorm of its projection onto data space
         x=x+alpha.*p;          #We update our initial model guess multiplying alpha by the original model error (?)
         s=s-alpha.*q;          #We update the data error subtracting alpha*model error projected on data
         r= A'*s;               #We project the updated data error into the model space.
         new=r'*r;
         beta=new/old;          #Ratio between the new and old norm of the model error (?)
         old=new;               #Variable update
         p = r.+(beta.*p);      #Updating the model error by advancing using the ratio between new and old norm.
         q= A*p;
         println("Iteration",k)                #Taking the new model error and projecting it into the data space.
        #error[k]=new
        end
    return x
end

function Derivatives(M,n)
    #n=2
    (a,b)=size(M);
    if n==1
        global Dz=zeros(a-1,a); global Dh=zeros(b,b-1);
    #                                      1ST DERIVATIVE
        Dh[diagind(Dh,0)].=1; Dh[diagind(Dh,-1)].=-1; Dz[diagind(Dz,0)].=1; Dz[diagind(Dz,1)].=-1
    elseif n==2
        global Dz=zeros(a-2,a); global Dh=zeros(b,b-2);
    #                                      2ND DERIVATIVE
        Dh[diagind(Dh,0)].=1; Dh[diagind(Dh,-1)].=-2; Dh[diagind(Dh,-2)].=1; 
        Dz[diagind(Dz,0)].=1; Dz[diagind(Dz,1)].=-2; Dz[diagind(Dz,2)].=1
    end
    ##########################################################################################
    m=vec(M);
    Iz=Matrix(I,length(M[1,:]),length(M[1,:]));
    Ih=Matrix(I,length(M[:,1]),length(M[:,1]))
    vdm=Dz*M;
    hdm=M*Dh;
    BDz=kron(Iz,Dz);
    BDh=kron(Dh',Ih);
    if n==1
        vdvec=reshape(BDz*m,length(M[:,1])-1,length(M[1,:]));
        hdvec=reshape(BDh*m,length(M[1,:])-1,length(M[:,1]));
    elseif n==2
        vdvec=reshape(BDz*m,length(M[:,1])-2,length(M[1,:]));
        hdvec=reshape(BDh*m,length(M[:,1]),length(M[1,:])-2);
    end
    return  BDh,BDz 
end