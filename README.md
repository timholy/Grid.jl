# Grid operations for the Julia language

Continuous objects, such as functions or images, are frequently sampled on a regularly-spaced grid of points. The **Grid** module provides support for common operations on such objects. Currently, the two main operations are *interpolation* and *restriction/prolongation*. Restriction and prolongation are frequently used for solving partial differential equations by multigrid methods, but can also be used simply as fast, antialiased methods for two-fold resolution changes (e.g., in computing thumbnails).

## Installation

Within Julia, use the package manager:
```julia
Pkg.add("Grid")
```

## Usage

To use the Grid module, begin your code with

```julia
using Grid
```

### Interpolation

Let's define a quadratic function in one dimension, and evaluate it on an evenly-spaced grid of 5 points:
```julia
c = 2.3  # center
a = 8.1  # quadratic coefficient
o = 1.6  # vertical offset
qfunc = x -> a*(x-c).^2 + o
xg = Float64[1:5]
y = qfunc(xg)
```

From these evaluated points, let's define an *interpolation grid*:
```julia
yi = InterpGrid(y, BCnil, InterpQuadratic)
```
The last two arguments will be described later. Note that only `y` is supplied; as with Julia's arrays, the x-coordinates implicitly start at 1 and increase by 1 with each grid point. It's easy to check that `yi[i]` is equal to `y[i]`, within roundoff error:
```julia
julia> y[3]
5.569000000000003

julia> yi[3]
5.569000000000003
```
However, unlike `y`, `yi` can also be evaluated at off-grid points:
```julia
julia> yi[1.9]
2.8959999999999995

julia> qfunc(1.9)
2.8959999999999995
```
It's also possible to evaluate the *slope* of the interpolation function:
```julia
julia> v,g = valgrad(yi, 4.25)
(32.40025000000001,31.590000000000003)

julia> 2*a*(4.25-c)
31.59
```
or the *second-order derivative* (in multidimensional cases, the *Hessian matrix*):
```julia
julia> v,g,h = valgradhess(yi, 4.25)
(32.40025,31.59000000000001,16.200000000000017)

julia> 2a
16.2
```

While these examples are one-dimensional, you can do interpolation on arrays of any dimensionality.

The last two parameters to `InterpGrid` specify the *boundary conditions* (what happens near, or beyond, the edges of the grid) and the *interpolation order*. The choices are specified below:

<table>
  <tr>
    <td>mode</td> <td>Meaning</td>
  </tr>
  <tr>
    <td>BCnil</td> <td>generates an error when points beyond the grid edge are needed</td>
  </tr>
  <tr>
    <td>BCnan</td> <td>generate NaN when points beyond the grid edge are needed</td>
  </tr>
  <tr>
    <td>BCreflect</td> <td>reflecting boundary conditions</td>
  </tr>
  <tr>
    <td>BCperiodic</td> <td>periodic boundary conditions</td>
  </tr>
  <tr>
    <td>BCnearest</td> <td>when beyond the edge, use the value at the closest interior point</td>
  </tr>
  <tr>
    <td>BCfill</td> <td>produce a specified value when beyond the edge</td>
  </tr>
</table>

The interpolation order can be one of the following:

<table>
  <tr>
    <td>InterpNearest</td> <td>nearest-neighbor (one-point) interpolation</td>
  </tr>
  <tr>
    <td>InterpLinear</td> <td>piecewise linear (two-point) interpolation (bilinear in two dimensions, etc.)</td>
  </tr>
  <tr>
    <td>InterpQuadratic</td> <td>quadratic (three-point) interpolation</td>
  </tr>
</table>

Note that "quadratic interpolation" is technically "non-interpolating", meaning that the coefficients of the interpolating polynomial are not the function values at the grid points. `InterpGrid` solves the tridiagonal system of equations for you, so in simple cases you do not need to worry about such details. `InterpQuadratic` is the lowest order of interpolation that yields a continuous gradient, and hence is suitable for use in gradient-based optimization.

In `d` dimensions, interpolation references `n^d` grid points, where `n` is the number of grid points used in one dimension. `InterpQuadratic` corresponds to `n=3`, and cubic spline interpolation would correspond to `n=4`. Consequently, in higher dimensions quadratic interpolation can be a significant savings relative to cubic spline interpolation.

#### Low-level interface

It should be noted that, in addition to the high-level `InterpGrid` interface, **Grid** also has lower-level interfaces. Users who need to extract values from multi-valued functions (e.g., an RGB image, which has three colors at each position) can achieve significant savings by using this low-level interface. The main cost of interpolation is computing the coefficients, and by using the low-level interface you can do this just once at each x location and use it for each color channel.

Here's an example using the low-level interface, starting from the one-dimensional quadratic example above:
```julia
y = qfunc(xg)
# Do the following once
interp_invert!(y, BCnan, InterpQuadratic)   # solve for generalized interp. coefficients
ic = InterpGridCoefs(y, InterpQuadratic)    # prepare for interpolation on this grid

# Do the following for each evaluation point
set_position(ic, BCnil, true, true, [1.8])  # set position to x=1.8, computes the coefs
v = interp(ic, y)                           # extract the value
# Do this if you want the slope at the same point
set_gradient_coordinate(ic, 1)              # change coefs to calc gradient along coord 1
g = interp(ic, y)                           # extract the gradient
# Do this to evaluate the Hessian at that point
set_hessian_coordinate(ic, 1, 2)            # change coefs to calc hessian element H[1,2], i.e. d2/dxdy
h = interp(ic, y)                           # extract hessian element
```
If this were an RGB image, you could call `interp` once for the red color channel, once for the green, and once for the blue, with just one call to `set_position`.

### Restriction and prolongation

Suppose you have an RGB image stored in an array `img`, where the third dimension is of length 3 and specifies the color. You can create a 2-fold smaller version of the image using `restrict`:
```julia
julia> size(img)
(1920,1080,3)

julia> imgr = restrict(img, [true,true,false]);

julia> size(imgr)
(961,541,3)
```
The second argument to `restrict` specifies which dimensions should be down-sampled.

Notice that the sizes are not precisely 2-fold smaller; this is because restriction is technically defined as the adjoint of prolongation, and prolongation interpolates (linearly) at intermediate points. For prolongation, you also have to supply the desired size:
```julia
julia> img2 = prolong(imgr, [size(img)...]);

julia> size(img2)
(1920,1080,3)
```
If a given dimension has size `n`, then the prolonged dimension can be either of size `2n-2` (if you want it to be even) or `2n-1` (if you want it to be odd). For odd-sized outputs, the interpolation is at half-grid points; for even-sized outputs, all output values are interpolated, at 1/4 and 3/4 grid points. Having both choices available makes it possible to apply `restrict` to arrays of any size.

If you plan multiple rounds of restriction, you can get the "schedule" of sizes from the function `restrict_size`:
```
julia> pyramid = restrict_size(size(img), [true true true true; true true true true; false false false false])
3x4 Int64 Array:
 961  481  241  121
 541  271  136   69
   3    3    3    3
```

Restriction and prolongation are extremely fast operations, because the coefficients can be specified in advance. For floating-point data types, this implementation makes heavy use of the outstanding performance of BLAS's `axpy`.

## Credits

Timothy E. Holy, 2012
