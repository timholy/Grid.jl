using Grid, Base.Test
include(Pkg.dir("Grid", "src/newinterp.jl"))

function runinterp(n, G, x)
    s = 0.0
    for j = 1:n
        for i = 1:length(x)
            s += G[x[i]]
        end
    end
    s
end

function runinterp(n, G, x, y)
    s = 0.0
    for j = 1:n
        for i = 1:length(x)
            s += G[x[i], y[i]]
        end
    end
    s
end

A1 = float([1,2,3,4])
A2 = rand(4,4)
x = rand(1.0:3.999, 10^6)
y = rand(1.0:3.999, 10^6)

for BC in (BCnil, BCnan, BCna, BCperiodic, BCnearest)
    println(BC)

    G = InterpGrid(A1, BC, InterpLinear)
    Gnew = InterpNew.InterpGridNew{Float64,1,BC,InterpLinear}(A1, 0.0);

    runinterp(1, G, [1.1])
    runinterp(1, Gnew, [1.1])
    @time s = runinterp(10, G, x)
    @time snew = runinterp(10, Gnew, x)
    @test_approx_eq s snew


    G = InterpGrid(A2, BC, InterpLinear)
    Gnew = InterpNew.InterpGridNew{Float64,2,BC,InterpLinear}(A2, 0.0);

    runinterp(1, G, [1.1], [1.2])
    runinterp(1, Gnew, [1.1], [1.2])
    @time s = runinterp(10, G, x, y)
    @time snew = runinterp(10, Gnew, x, y)
    @test_approx_eq s snew
end
