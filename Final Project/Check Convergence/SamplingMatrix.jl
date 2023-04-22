
function SamplingVector(in::Vector)

    cutoff = 1e-10
    i = 1
    
    #    n=size(in)
    #   in=reshape(in,n[1],:)
    wd = zeros(Int,length(in))
    #    n2=size(in,2)
    for i = 1 : length(in)
        a =(in[i])^2;
        if (abs(a) > cutoff)
            wd[i] = 1;
        end
    end
    return wd;
end



function SamplingMatrix(in::Vector;type="seismic")

    #r=SamplingVector(in::Vector);

    r=SamplingVector(in);

    if type == "seismic"
        
       # r=SamplingVector(in);

        M= count(i->(i == 0),r) # dead receivers
        N= count(i->(i == 1),r) # active receivers
        Nr=length(r);
        I= diagm(ones(Nr))#.*r;
        R= I.*r;
        R= R[vec(mapslices(col -> any(col .!= 0), R, dims = 2)), :];

        if size(R) == (N,Nr)
            return R
        end
    
    else

        M= count(i->(i == 0),r) # dead receivers
        N= count(i->(i == 1),r) # active receivers
        Nr=length(r);

        R=randn(Float64,(N,Nr));

        return R

    end
end


