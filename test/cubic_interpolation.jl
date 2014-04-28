# Cubic interpolation
a = 4.3
b = 1.7
c = 6.2
d = 5.9
cfunc = x -> a*x.^3 .+ b*x.^2 .+ c*x .+ d
cgrad = x -> 3a*x.^2 .+ 2b*x .+ c
chess = x -> 6a*x .+ 2b
xg = Float64[1:8]
y = cfunc(xg)

# Low-level interface
yc = copy(y)
interp_invert!(yc,BCnan,InterpCubic)
ic = InterpGridCoefs(yc,InterpCubic)

x = 3.7
set_position(ic, BCnil, true, true, [x])
yi = interp(ic,yc)
@test_approx_eq_eps cfunc(x) yi 0.01*(abs(cfunc(x))+abs(yi))
set_gradient_coordinate(ic, 1)
gi = interp(ic, yc)
@test_approx_eq_eps cgrad(x) gi 0.1*(abs(cgrad(x))+abs(yi))
set_hessian_coordinate(ic, 1, 1)
hi = interp(ic, yc)
@test_approx_eq_eps chess(x) hi 0.1*(abs(chess(x))+abs(yi))

# High-level interface
yi = InterpGrid(y, BCnil, InterpCubic)
@test_approx_eq_eps yi[x] cfunc(x) 0.01*(abs(cfunc(x))+abs(yi[x]))
v,g = valgrad(yi,x)
@test_approx_eq_eps v cfunc(x) 0.01*(abs(cfunc(x))+abs(yi[x]))
@test_approx_eq_eps g cgrad(x) 0.01*(abs(cgrad(x))+abs(yi[x]))
v,g,h = valgradhess(yi,x)
@test_approx_eq_eps v cfunc(x) 0.01*(abs(cfunc(x))+abs(yi[x]))
@test_approx_eq_eps g cgrad(x) 0.01*(abs(cgrad(x))+abs(yi[x]))
@test_approx_eq_eps h chess(x) 0.01*(abs(chess(x))+abs(yi[x]))

# Test derivatives with other boundary conditions
func = x -> yi[x]
gnum = derivative_numer(func,x)
hnum = derivative2_numer(func,x)

for BC in (BCnan, BCna)
    yi = InterpGrid(y, BC, InterpCubic)
    func = x -> yi[x]
    v,g = valgrad(yi,x)
    gnum = derivative_numer(func,x) 
    @test_approx_eq_eps g gnum cbrt(eps())*(abs(g)+abs(gnum))
    v,g,h = valgradhess(yi,x)
    gnum = derivative_numer(func,x)
    hnum = derivative2_numer(func,x)
    @test_approx_eq_eps g gnum cbrt(eps())*(abs(g)+abs(gnum))
    @test_approx_eq_eps h hnum cbrt(eps())*(abs(h)+abs(hnum))
end

# Test derivatives of 2D interpolation

y = rand(7,8)
yi = InterpGrid(y,BCnan,InterpCubic)
x = [3.4,6.1]
v,g = valgrad(yi,x...)
v2,g2,h2 = valgradhess(yi,x...)
func = (x1,x2) -> yi[x1,x2]

gnum = derivative_numer(func, tuple(x...), 1)
@test_approx_eq_eps g[1] gnum cbrt(eps())*(abs(g[1])+abs(gnum))
@test_approx_eq_eps g2[1] gnum cbrt(eps())*(abs(g2[1])+abs(gnum))

gnum = derivative_numer(func, tuple(x...), 2)
@test_approx_eq_eps g[2] gnum cbrt(eps())*(abs(g[2])+abs(gnum))
@test_approx_eq_eps g2[2] gnum cbrt(eps())*(abs(g2[2])+abs(gnum))

hnum = derivative2_numer(func, tuple(x...), (1, 1))
@test_approx_eq_eps h2[1,1] hnum sqrt(sqrt(eps()))*(abs(h2[1,1])+abs(hnum))
hnum = derivative2_numer(func, tuple(x...), (1, 2))
@test_approx_eq_eps h2[1,2] hnum sqrt(sqrt(eps()))*(abs(h2[1,2])+abs(hnum))
@test_approx_eq_eps h2[2,1] hnum sqrt(sqrt(eps()))*(abs(h2[2,1])+abs(hnum))
hnum = derivative2_numer(func, tuple(x...), (2, 2))
@test_approx_eq_eps h2[2,2] hnum sqrt(sqrt(eps()))*(abs(h2[2,2])+abs(hnum))
