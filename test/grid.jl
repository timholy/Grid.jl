using Grid, Base.Test

include("derivative_numer.jl")

## Interpolation ##
# Low-level components
dims = (4, 6)
z = zeros(dims)
ic = InterpGridCoefs(z, InterpQuadratic)
x = [2,3.5]
set_position(ic, BCnearest, true, x)
@assert ic.coord1d == Vector{Int}[[1,2,3],[3,4,5]]
@assert ic.coef1d == Vector{Float64}[[1/8,3/4,1/8],[1/2,1/2,0]]
@assert ic.wrap == false
z[ic.offset.+ic.offset_base.+1] = ic.coef
@assert z == [0 0 1/16 1/16 0 0; 0 0 3/8 3/8 0 0; 0 0 1/16 1/16 0 0; 0 0 0 0 0 0]
# Test that the generic coef algorithm gives the same result as the
# dimension-specialized versions
for n_dims = 1:4
    dims = fill(3, n_dims)
    s = similar(dims)
    s[1] = 1
    for idim = 1:n_dims-1
        s[idim+1] = dims[idim]*s[idim]
    end
    ic = InterpGridCoefs(Float64, InterpQuadratic, dims, s)
    x = rand(n_dims) .+ 1.5
    x[x .== 2.5] = 2.49999
    set_position(ic, BCnearest, true, x)
    @assert ic.wrap == false
    @assert abs(ic.coef[iceil(length(ic.coef)/2)] .- prod(3/4.-(x.-2).^2)) < eps()
    index = ic.offset.+ic.offset_base.+1
    indices_generic = similar(index)
    coefs_generic = similar(ic.coef)
    Grid.interp_index_coef_generic(indices_generic, coefs_generic, ic.coord1d, ic.coef1d, s)
    @assert index == indices_generic
    @assert ic.coef == coefs_generic
end

# On-grid values
A = randn(4,10)
const EPS = sqrt(eps())
for bc in (BCnil, BCnan, BCna, BCreflect, BCperiodic, BCnearest, BCfill)
    for it in (InterpNearest, InterpLinear, InterpQuadratic, InterpCubic)
        # skip boundary conditions that aren't implemented for cubic yet
        if it == InterpCubic && !(bc <: Union(BCnan,BCna,BCnil))
            continue
        end
        ig = bc == BCfill ? InterpGrid(A, 0.0, it) : InterpGrid(A, bc, it)
        for i = 1:size(A,1)
            for j = 1:size(A,2)
                @test_approx_eq_eps ig[i,j] A[i,j] EPS
            end
        end
        v = ig[1:size(A,1), 1:size(A,2)]
        @assert all(abs(v - A) .< EPS)
    end
end
A = randn(4,5,4)
for bc in (BCnil, BCnan, BCna, BCreflect, BCperiodic, BCnearest, BCfill)
    for it in (InterpNearest, InterpLinear, InterpQuadratic, InterpCubic)
        # skip boundary conditions that aren't implemented for cubic yet
        if it == InterpCubic && !(bc <: Union(BCnan,BCna,BCnil))
            continue
        end
        ig = bc == BCfill ? InterpGrid(A, 0.0, it) : InterpGrid(A, bc, it)
        for k = 1:size(A,3), j = 1:size(A,2), i = 1:size(A,1)
            @assert abs(ig[i,j,k] - A[i,j,k]) < EPS
        end
        v = ig[1:size(A,1), 1:size(A,2), 1:size(A,3)]
        @assert all(abs(v - A) .< EPS)
    end
end
A = randn(4,5,4,3)
for bc in (BCnil, BCnan, BCna, BCreflect, BCperiodic, BCnearest, BCfill)
    for it in (InterpNearest, InterpLinear, InterpQuadratic, InterpCubic)
        # skip boundary conditions that aren't implemented for cubic yet
        if it == InterpCubic && !(bc <: Union(BCnan,BCna,BCnil))
            continue
        end
        ig = bc == BCfill ? InterpGrid(A, 0.0, it) : InterpGrid(A, bc, it)
        for l = 1:size(A,4), k = 1:size(A,3), j = 1:size(A,2), i = 1:size(A,1)
            @assert abs(ig[i,j,k,l] - A[i,j,k,l]) < EPS
        end
        v = ig[1:size(A,1), 1:size(A,2), 1:size(A,3), 1:size(A,4)]
        @assert all(abs(v - A) .< EPS)
    end
end

A = float([1:4])
for it in (InterpNearest, InterpLinear, InterpQuadratic)
    ig = InterpGrid(A, BCreflect, it)
    y = ig[-3:8]
    @assert all(abs(y-A[[4:-1:1,1:4,4:-1:1]]) .< EPS)
end
for it in (InterpNearest, InterpLinear, InterpQuadratic)
    ig = InterpGrid(A, BCnearest, it)
    y = ig[0:6]
    @assert all(abs(y-A[[1,1:4,4,4]]) .< EPS)
end

# Quadratic interpolation
c = 2.3
a = 8.1
o = 1.6
qfunc = x -> a*(x.-c).^2 .+ o
xg = Float64[1:5]
y = qfunc(xg)
yc = copy(y)
interp_invert!(yc, BCnan, InterpQuadratic)
ic = InterpGridCoefs(yc, InterpQuadratic)

x = 1.8
set_position(ic, BCnil, true, true, [x])
yi = interp(ic, yc)
@assert abs(yi - qfunc(x)) < 100*eps()
set_gradient_coordinate(ic, 1)
g = interp(ic, yc)
@assert abs(g - 2*a*(x-c)) < 100*eps()
set_hessian_coordinate(ic, 1, 1)
h = interp(ic, yc)
@test_approx_eq_eps h 2a 100*eps()

# High-level interface
yi = InterpGrid(y, BCnil, InterpQuadratic)
@assert abs(yi[x] - qfunc(x)) < 100*eps()
v,g = valgrad(yi, x)
@assert abs(v - qfunc(x)) < 100*eps()
@assert abs(g - 2*a*(x-c)) < 100*eps()
v2, g2, h2 = valgradhess(yi, x)
@test_approx_eq_eps v2 qfunc(x) 100*eps()
@test_approx_eq_eps g2 2a*(x-c) 100*eps()
@test_approx_eq_eps h2 2a 100*eps()

# Test derivatives with other boundary conditions
Eps = sqrt(eps())
func = x -> yi[x]
gnum = derivative_numer(func, x)
hnum = derivative2_numer(func, x)
@assert abs(g-gnum) < Eps*(abs(g)+abs(gnum))
@test_approx_eq_eps g2 gnum Eps*(abs(g2)+abs(gnum))
@test_approx_eq_eps h2 hnum cbrt(eps())*(abs(h2)+abs(hnum))

for BC in (BCnan, BCna, BCnearest, BCperiodic, BCreflect, 0)
    yi = InterpGrid(y, BC, InterpQuadratic)
    func = x -> yi[x]
    v,g = valgrad(yi, x)
    gnum = derivative_numer(func, x)
    @assert abs(g-gnum) < Eps*(abs(g)+abs(gnum))
    v2,g2,h2 = valgradhess(yi, x)
    hnum = derivative2_numer(func, x)
    @test_approx_eq_eps g2 gnum Eps*(abs(g2)+abs(gnum))
    @test_approx_eq_eps h2 hnum cbrt(eps())*(abs(h2)+abs(hnum))
end

# Test derivatives of 2d interpolation
y = rand(7,8)
yi = InterpGrid(y, BCreflect, InterpQuadratic)
x = [2.2, 3.1]
v,g = valgrad(yi, x...)
v2,g2,h2 = valgradhess(yi, x...)
func = (x1,x2) -> yi[x1,x2]
gnum = derivative_numer(func, tuple(x...), 1)
@assert abs(g[1]-gnum) < Eps*(abs(g[1])+abs(gnum))
@test_approx_eq_eps g2[1] gnum Eps*(abs(g2[1])+abs(gnum))
gnum = derivative_numer(func, tuple(x...), 2)
@test_approx_eq_eps g[2]  gnum Eps*(abs(g[2])+abs(gnum))
@test_approx_eq_eps g2[2] gnum Eps*(abs(g2[2])+abs(gnum))
hnum = derivative2_numer(func, tuple(x...), (1, 1))
@test_approx_eq_eps h2[1,1] hnum 1e-5
hnum = derivative2_numer(func, tuple(x...), (1, 2))
@test_approx_eq_eps h2[1,2] hnum 1e-5
@test_approx_eq_eps h2[2,1] hnum 1e-5
hnum = derivative2_numer(func, tuple(x...), (2, 2))
@test_approx_eq_eps h2[2,2] hnum 1e-5

# Nearest-neighbor value and gradient
y = [float(2:6)]
ig = InterpGrid(y, BCnan, InterpNearest)
x = 1.8
v, g = valgrad(ig, x)
@assert v == 3
@assert g == 0
@assert ig[1.8:5.4] == [3:6]

# Linear value and gradient
y = [float(2:6)]
ig = InterpGrid(y, BCnan, InterpLinear)
x = 1.8
v, g = valgrad(ig, x)
@assert abs(v - 2.8) < Eps
@assert abs(g - 1.0) < Eps
@assert all(abs(ig[1.8:5.4] - [2.8:5.8]) .< Eps)

#### Interpolation on irregularly-spaced grids ####
x = [100.0,110.0,150.0]
y = rand(3)
iu = InterpIrregular(x, y, -200, InterpNearest)
@assert iu[99] == -200
@assert iu[101] == y[1]
@assert iu[106] == y[2]
@assert iu[149] == y[3]
@assert iu[150.1] == -200
iu = InterpIrregular(x, y, BCna, InterpLinear)
@assert isnan(iu[99])
@assert abs(iu[101] - (0.9*y[1] + 0.1*y[2])) < Eps
@assert abs(iu[106] - (0.4*y[1] + 0.6*y[2])) < Eps
@assert abs(iu[149] - (y[2]/40 + (39/40)*y[3])) < Eps
@assert isnan(iu[150.1])
iu = InterpIrregular(x, y, BCnil, InterpLinear)
@test_throws BoundsError isnan(iu[99])
@assert abs(iu[101] - (0.9*y[1] + 0.1*y[2])) < Eps
@assert abs(iu[106] - (0.4*y[1] + 0.6*y[2])) < Eps
@assert abs(iu[149] - (y[2]/40 + (39/40)*y[3])) < Eps
@test_throws BoundsError isnan(iu[150.1])
iu = InterpIrregular(x, y, BCnearest, InterpLinear)
@assert iu[99] == y[1]
@assert abs(iu[101] - (0.9*y[1] + 0.1*y[2])) < Eps
@assert abs(iu[106] - (0.4*y[1] + 0.6*y[2])) < Eps
@assert abs(iu[149] - (y[2]/40 + (39/40)*y[3])) < Eps
@assert iu[150.1] == y[3]


#### Restrict/prolong ####
# 1d
a = zeros(5)
a[3] = 1
p = prolong(a, [9])
@assert all(p .== [0,0,0,0.5,1,0.5,0,0,0])
r = restrict([0,0,0,0,1.0,0,0,0,0], 1)
@assert all(r .== [0,0,1,0,0])
r = restrict([0,1.0,0,0,0,0,0,0,0], 1)
@assert all(r .== [0.5,0.5,0,0,0])
r = restrict(p, 1)
@assert all(r .== [0,0.25,1.5,0.25,0])
p = prolong(a, [8])
@assert all(p .== [0,0,0.25,0.75,0.75,0.25,0,0])
a[3] = 0
a[1] = 1
p = prolong(a, [9])
@assert all(p .== [1,0.5,0,0,0,0,0,0,0])
p = prolong(a, [8])
@assert all(p .== [0.75,0.25,0,0,0,0,0,0])
a = [0,0,1.0,0]
p = prolong(a, [7])
@assert all(p .== [0,0,0,0.5,1,0.5,0])
p = prolong(a, [6])
@assert all(p .== [0,0,0.25,0.75,0.75,0.25])
r = restrict([0,1.0,0,0,0,0,0,0], 1)
@assert all(r .== [0.25,0.75,0,0,0])
r = restrict([0,0,1.0,0,0,0,0,0], 1)
@assert all(r .== [0,0.75,0.25,0,0])
# Check with non-BLAS types
a = Vector{Float64}[[0.0],[0.0],[1.0],[0.0],[0.0]]
p = prolong(a, [9])
@assert all(p .== Vector{Float64}[[0.0],[0.0],[0.0],[0.5],[1.0],[0.5],[0.0],[0.0],[0.0]])
p = prolong(a, [8])
@assert all(p .== Vector{Float64}[[0.0],[0.0],[0.25],[0.75],[0.75],[0.25],[0.0],[0.0]])
a = Vector{Float64}[[0],[0],[1.0],[0]]
p = prolong(a, [7])
@assert all(p .== Vector{Float64}[[0],[0],[0],[0.5],[1],[0.5],[0]])
p = prolong(a, [6])
@assert all(p .== Vector{Float64}[[0],[0],[0.25],[0.75],[0.75],[0.25]])
# 2d
a = zeros(5,5)
a[3,3] = 1
p = prolong(a, 1, 9)
@assert all(p .== [zeros(9,2) [0,0,0,0.5,1,0.5,0,0,0] zeros(9,2)])
p = prolong(a, 2, 9)
@assert all(p .== [zeros(9,2) [0,0,0,0.5,1,0.5,0,0,0] zeros(9,2)]')
p = prolong(a, 1, 8)
@assert all(p .== [zeros(8,2) [0,0,0.25,0.75,0.75,0.25,0,0] zeros(8,2)])
p = prolong(a, 2, 8)
@assert all(p .== [zeros(8,2) [0,0,0.25,0.75,0.75,0.25,0,0] zeros(8,2)]')
r = restrict(a, [true,false])
@assert all(r .== [zeros(3,2) [0,1.0,0] zeros(3,2)])
r = restrict(a, [false,true])
@assert all(r .== [zeros(3,2) [0,1.0,0] zeros(3,2)]')
a = zeros(6,6)
a[3,3] = 1
r = restrict(a, [true,false])
@assert all(r .== [zeros(4,2) [0,0.75,0.25,0] zeros(4,3)])
r = restrict(a, [false,true])
@assert all(r .== [zeros(4,2) [0,0.75,0.25,0] zeros(4,3)]')
# Non-BLAS
a = fill([0.0], 5, 5)
a[3,3] = [1.0]
p = prolong(a, 1, 9)
ptest = fill([0.0], 9, 5)
ptest[6,3] = ptest[4,3] = [0.5]
ptest[5,3] = [1.0]
@assert all(p .== ptest)
p = prolong(a, 2, 9)
@assert all(p .== ptest')
r = restrict(a, [true,false])
rtest = fill([0.0], 3, 5)
rtest[2,3] = [1.0]
@assert all(r .== rtest)
r = restrict(a, [false,true])
@assert all(r .== rtest')
a = fill([0.0], 5, 5)
a[2,3] = [1.0]
rtest = fill([0.0], 3, 5)
rtest[1,3] = rtest[2,3] = [0.5]
r = restrict(a, [true,false])
@assert all(r .== rtest)
a = fill([0.0], 4, 4)
a[2,3] = [1.0]
r = restrict(a, [true,false])
rtest = fill([0.0], 3, 4)
rtest[1,3] = [0.25]
rtest[2,3] = [0.75]
@assert all(r .== rtest)
r = restrict(a, [false,true])
rtest = fill([0.0], 4, 3)
rtest[2,2] = [0.75]
rtest[2,3] = [0.25]
@assert all(r .== rtest)
