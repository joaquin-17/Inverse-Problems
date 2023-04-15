# This function generates the ricker wavelet and it is equivalent to Scipy.signal.ricker functiondef ricker(N, a):
    """
    Ricker function
    """


    function RickerWavelet(N::Int64,width::Float64)
        A = 2 /(sqrt(3 * width)*(pi^(1/4)))
        wsq = width^(2)
        vec=collect(-1*(N-1)/2:1:(N-1)/2)
        xsq = vec.^(2)
        modd = (1 .- xsq / wsq);
        gauss = exp.(-1*xsq /(2 * wsq))
        total = A.*modd.* gauss
        
        return total

    end
