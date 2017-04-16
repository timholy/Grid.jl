using Grid, Base.Test
include(Pkg.dir("Grid", "src/newinterp.jl"))

## Construct ground-truth values for 1d interpolation
A1 = float([1,2,3,4])
xpos = [-1.5:0.1:6.5]
inbounds = 1 .<= xpos .<= length(A1)
A1inbounds = xpos[inbounds]
A1nil = fill(Inf, length(xpos)); A1nil[inbounds] = A1inbounds  # use Inf as a sentinel for BoundsError
A1nan = fill(NaN, length(xpos)); A1nan[inbounds] = A1inbounds
A1na = fill(NaN, length(xpos)); A1na[inbounds] = A1inbounds; A1na[0 .< xpos .< 1] = 1; A1na[4 .< xpos .< 5] = 4
A1reflect = zeros(length(xpos)); A1reflect[inbounds] = A1inbounds; A1reflect[0 .<= xpos .<= 1] = 1; A1reflect[4 .<= xpos .<= 5] = 4
l = findfirst(xpos .>= 0); A1reflect[1:l] = A1inbounds[l:-1:1]; l = findfirst(xpos .>= 5); A1reflect[l:end] = A1inbounds[end:-1:length(A1reflect)-l+1]
xirange = ifloor(minimum(xpos))-1:iceil(maximum(xpos))+1
Aiper = A1[Int[mod(x-1,length(A1))+1 for x in xirange]]
A1periodic = similar(A1na)
for i = 1:length(xpos)
    x = xpos[i]
    ix = ifloor(x)
    j = find(xirange .== ix)[1]
    fx = x - ix
    A1periodic[i] = (1-fx)*Aiper[j]+fx*Aiper[j+1]
end
A1nearest = zeros(length(xpos)); A1nearest[inbounds] = A1inbounds; A1nearest[xpos .< 1] = 1; A1nearest[4 .< xpos] = 4
A1fill = zeros(length(xpos)); A1fill[inbounds] = A1inbounds
# Plot the values, to make sure we've done it right
if isdefined(Main, :Gadfly)
    println("Plotting")
    Aall = vcat(
        DataFrame(x = xpos, y = A1nil, t = "BCnil"),
        DataFrame(x = xpos, y = A1nan, t = "BCnan"),
        DataFrame(x = xpos, y = A1na,  t = "BCna"),
        DataFrame(x = xpos, y = A1reflect, t = "BCreflect"),
        DataFrame(x = xpos, y = A1periodic, t = "BCperiodic"),
        DataFrame(x = xpos, y = A1nearest, t = "BCnearest"),
        DataFrame(x = xpos, y = A1fill, t = "BCfill"))
    set_default_plot_size(30cm, 7.5cm)
    plot(Aall, x = :x, y = :y, xgroup = :t, Geom.subplot_grid(Geom.line))
end


for (BC,correct) in ((BCnil,A1nil), (BCnan,A1nan), (BCna,A1na), (BCreflect, A1reflect), (BCperiodic, A1periodic), (BCnearest, A1nearest), (BCfill,A1fill))
    println(BC)
    G = InterpNew.InterpGridNew{Float64,1,BC,InterpLinear}(A1, 0.0);
    for i = 1:length(xpos)
        x = xpos[i]
        y = correct[i]
        if !isinf(y)
            try
                @test_approx_eq y G[x]
            catch err
                @show x, y
                rethrow(err)
            end
        else
            @test_throws BoundsError G[x]
        end
    end
end
