using Grid
using Base.Test

x = -1.0:0.1:1.0
z = sin(x)

cg = CoordInterpGrid(x,z,BCnil,InterpQuadratic)

@test abs(cg[0.0] - sin(0.0)) < 100*eps()
@test abs(cg[0.3] - sin(0.3)) < 100*eps()

v,g = valgrad(cg,0.0)
@test abs(g - cos(0.0)) < 0.001

v,g = valgrad(cg,0.3)
@test abs(g - cos(0.3)) < 0.001



y = -2.0:0.1:2.0
z = Float64[sin(i+j) for i in x, j in y]

@test_throws DimensionMismatch cg = CoordInterpGrid((y,x),z,BCnil,InterpQuadratic)
cg = CoordInterpGrid((x,y),z,BCnil,InterpQuadratic)

@test abs(cg[0.0,0.0] - sin(0.0)) < 0.0001
@test abs(cg[0.0,0.3] - sin(0.3)) < 0.0001
@test abs(cg[0.3,0.0] - sin(0.3)) < 0.0001

v,g = valgrad(cg,0.0,0.0)
@test abs(g[1] - cos(0.0)) < 0.001
@test abs(g[2] - cos(0.0)) < 0.001

v = valgrad(g,cg,0.3,0.0)
@test abs(g[1] - cos(0.3)) < 0.001
@test abs(g[2] - cos(0.3)) < 0.001

# Issue #29
xg = -10:.5:10
yg = -3:.15:3.5
zg = Float64[sin(x)*cos(y) for x in xg, y in yg]
ig = CoordInterpGrid((xg,yg),zg, 0, InterpLinear)
@test size(ig) == size(zg)
@test ig[xg[1],yg[1]] == zg[1,1]
