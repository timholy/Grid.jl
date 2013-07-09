module Grid
import Base: done, next, start, convert, eltype, getindex, isvalid, ndims, show, size

export
# Types
    BoundaryCondition,
    BCnil,
    BCnan,
    BCna,
    BCreflect,
    BCperiodic,
    BCnearest,
    BCfill,
    Counter,
    InterpGrid,
    InterpGridCoefs,
    InterpIrregular,
    InterpType,
    InterpNearest,
    InterpLinear,
    InterpQuadratic,
# Functions
    filledges!,
    interp,
    interp_invert!,
    npoints,
    pad1,
    prolong,
    prolongb,
    prolong_size,
    restrict,
    restrictb,
    restrict_size,
    restrict_extrap,
    set_gradient_coordinate,
    set_position,
    set_size,
    valgrad,
    wrap
    
#### Iterator ####
# An n-dimensional grid iterator. This allows you to do things in
# arbitrary dimensions that in fixed dimensions you might do with a
# comprehension.
#
# For example, for two dimensions the equivalent of
#    p = [ i*j for i = 1:m, j = 1:n ]
# is
#    sz = (m, n)
#    p = Array(Int, sz)
#    index = 1
#    for c = Counter(sz)
#      p[index] = prod(c)
#      index += 1
#    end
# However, with sz appropriately defined, this version works for
# arbitrary dimensions.

type Counter
    max::Vector{Int}
end
Counter(sz) = Counter(Int[sz...])

function start(c::Counter)
    N = length(c.max)
    state = ones(Int,N)
    if N > 0
        state[1] = 0 # because of start/done/next sequence, start out one behind
    end
    return state
end
function done(c::Counter, state)
    if isempty(state)
        return true
    end
    # we do the increment as part of "done" to make exit-testing more efficient
    state[1] += 1
    i = 1
    max = c.max
    while state[i] > max[i] && i < length(state)
        state[i] = 1
        i += 1
        state[i] += 1
    end
    state[end] > max[end]
end
next(c::Counter, state) = state, state

#### Boundary Conditions ####

abstract BoundaryCondition
type BCnil <: BoundaryCondition; end # ignore edges (error when invalid)
type BCnan <: BoundaryCondition; end  # NaN when over the edge
type BCna <: BoundaryCondition; end  # Normalize to available data (NaN when none)
type BCreflect <: BoundaryCondition; end # Reflecting boundary conditions
type BCperiodic <: BoundaryCondition; end # Periodic boundary conditions
type BCnearest <: BoundaryCondition; end # Return closest edge element
type BCfill <: BoundaryCondition; end # Use specified fill value

# Note: for interpolation, BCna is currently defined to be identical
# to BCnan. Other applications might define different behavior,
# particularly for filtering operations.

#needs_validating{BC<:BoundaryCondition}(::Type{BC}) = false
#needs_validating{BC<:Union(BCnil,BCnan,BCna)}(::Type{BC}) = true
isvalid{BC<:BoundaryCondition}(::Type{BC}, pos::Int, min::Int, max::Int) = true
isvalid{BC<:Union(BCnan,BCna)}(::Type{BC}, pos::Int, min::Int, max::Int) = min <= pos <= max
wrap{BC<:BoundaryCondition}(::Type{BC}, pos::Int, len::Int) = pos
wrap(::Type{BCreflect}, pos::Int, len::Int) = wraprefl(mod(pos-1, 2*len), len)
wraprefl(posrem::Int, len::Int) = posrem < len ? posrem+1 : 2*len-posrem
wrap(::Type{BCperiodic}, pos::Int, len::Int) = mod(pos-1, len) + 1
wrap{BC<:Union(BCnearest,BCfill)}(::Type{BC}, pos::Int, len::Int) = pos < 1 ? 1 : (pos > len ? len : pos)

#### Interpolation of evenly-spaced data ####

# This implements "generalized interpolation." See, for example,
# P. Thevenaz, T. Blu, and M. Unser (2000). "Interpolation Revisited."
# IEEE Transactions on Medical Imaging, 19: 739-758.

abstract InterpType
type InterpNearest <: InterpType; end
type InterpLinear <: InterpType; end
type InterpQuadratic <: InterpType; end
#type InterpCubic <: InterpType; end   # TODO

# This type manages temporary storage needed for efficient
# interpolation on a grid, so that once created it is possible to
# calculate many interpolated values without additional
# temporaries. It is designed to support both single- and multi-valued
# interpolation, and computation of the gradient with respect to the
# position of evaluation.
# An example of multi-valued interpolation is sub-pixel interpolation
# within an RGB image, and getting the 3 color values out.
type InterpGridCoefs{T<:FloatingPoint, IT<:InterpType}
    coord1d::Vector{Vector{Int}}  # for 1-d positions
    coef1d::Vector{Vector{T}}     # for 1-d coefficients
    gcoef1d::Vector{Vector{T}}    # for 1-d coefficients of the gradient
    c1d::Vector{Vector{T}}        # temp space for switch btw value & gradient
    dims::Vector{Int}
    strides::Vector{Int}
    offset::Vector{Int}  # the default N-d offset when there is no wrapping
    offset_base::Int
    index::Vector{Int}   # the alternative N-d index, needed when wrapping
    wrap::Bool           # which of these two indices to use
    valid::Bool          # some BCs return NaN when wrapping
    coef::Vector{T}      # the N-d coefficient of each neighbor (val & gradient)
end

#### InterpGrid ####
# This is the high-level interface. Use this for scalar-valued
# interpolation, and for calculation of the gradient with respect to
# the location of the evaluation point.
# For efficient multi-valued interpolation, use the low-level
# interface below. (getindex and valgrad provide examples of how to use the
# low-level interface.)
type InterpGrid{T<:FloatingPoint, N, BC<:BoundaryCondition, IT<:InterpType} <: AbstractArray{T,N}
    coefs::Array{T,N}
    ic::InterpGridCoefs{T, IT}
    x::Vector{T}
    fillval::T  # used only for BCfill (if ever)
end
function InterpGrid{T<:FloatingPoint, BC<:BoundaryCondition, IT<:InterpType}(A::Array{T}, ::Type{BC}, ::Type{IT})
    coefs = copy(A)
    interp_invert!(coefs, BC, IT, 1:ndims(A))
#    println(coefs)
    ic = InterpGridCoefs(coefs, IT)
    x = zeros(T, ndims(A))
    InterpGrid{T, ndims(A), BC, IT}(coefs, ic, x, nan(T))
end
function InterpGrid{T<:FloatingPoint, IT<:InterpType}(A::Array{T}, f::Number, ::Type{IT})
    coefs = pad1(A, f, 1:ndims(A))
    interp_invert!(coefs, BCnearest, IT, 1:ndims(A))
    ic = InterpGridCoefs(coefs, IT)
    x = zeros(T, ndims(A))
    InterpGrid{T, ndims(A), BCfill, IT}(coefs, ic, x, convert(T, f))
end


setx{T}(G::InterpGrid{T,1}, x::Real) = G.x[1] = x
function setx{T}(G::InterpGrid{T,2}, x::Real, y::Real)
    xG = G.x
    xG[1] = x
    xG[2] = y
end
function setx{T}(G::InterpGrid{T,3}, x::Real, y::Real, z::Real)
    xG = G.x
    xG[1] = x
    xG[2] = y
    xG[3] = z
end
function setx{T,N}(G::InterpGrid{T,N}, x::Real...)
    if length(x) != N
        error("Incorrect number of dimensions supplied")
    end
    xG = G.x
    for idim = 1:N
        xG[idim] = x[idim]
    end
end
# a version that corrects for the padding of BCfill types
setx{T}(G::InterpGrid{T,1,BCfill}, x::Real) = G.x[1] = x+1
function setx{T}(G::InterpGrid{T,2,BCfill}, x::Real, y::Real)
    xG = G.x
    xG[1] = x+1
    xG[2] = y+1
end
function setx{T}(G::InterpGrid{T,3,BCfill}, x::Real, y::Real, z::Real)
    xG = G.x
    xG[1] = x+1
    xG[2] = y+1
    xG[3] = z+1
end
function setx{T,N}(G::InterpGrid{T,N,BCfill}, x::Real...)
    if length(x) != N
        error("Incorrect number of dimensions supplied")
    end
    xG = G.x
    for idim = 1:N
        xG[idim] = x[idim]+1
    end
end

## Evaluation at single points
function _getindex{T}(G::InterpGrid{T})
    set_position(G.ic, boundarycondition(G), false, G.x)
    interp(G.ic, G.coefs)
end
function getindex(G::InterpGrid, x::Real...)
    setx(G, x...)
    _getindex(G)
end
function _valgrad{T,N}(g::Vector{T}, G::InterpGrid{T,N})
    if length(g) != N
        error("Wrong number of components for the gradient")
    end
    ic = G.ic
    coefs = G.coefs
    set_position(ic, boundarycondition(G), true, G.x)
    val = interp(ic, coefs)
    for idim = 1:N
        set_gradient_coordinate(ic, idim)
        g[idim] = interp(ic, coefs)
    end
    return val
end
function _valgrad{T}(G::InterpGrid{T,1})
    ic = G.ic
    coefs = G.coefs
    set_position(ic, boundarycondition(G), true, G.x)
    val = interp(ic, coefs)
    set_gradient_coordinate(ic, 1)
    g = interp(ic, coefs)
    return val, g
end
function valgrad{T}(G::InterpGrid{T,1}, x::Real)
    setx(G, x)
    _valgrad(G)
end
function valgrad{T}(G::InterpGrid{T}, x::Real...)
    setx(G, x...)
    g = Array(T, length(x))
    val = _valgrad(g, G)
    return val, g
end
function valgrad{T}(g::Vector{T}, G::InterpGrid{T}, x::Real...)
    setx(G, x...)
    _valgrad(g, G)
end

## Vectorized evaluation at multiple points
function getindex{T,R<:Real}(G::InterpGrid{T,1}, x::AbstractVector{R})
    n = length(x)
    v = Array(T, n)
    for i = 1:n
        setx(G, x[i])
        v[i] = _getindex(G)
    end
end
getindex{T,N,R<:Real}(G::InterpGrid{T,N}, x::AbstractVector{R}) = error("Linear indexing not supported")
function getindex{T,R<:Real}(G::InterpGrid{T,2}, x::AbstractVector{R}, y::AbstractVector{R})
    nx, ny = length(x), length(y)
    v = Array(T, nx, ny)
    for i = 1:nx
        for j = 1:ny
            setx(G, x[i], y[j])
            v[i,j] = _getindex(G)
        end
    end
end
function getindex{T,N,R<:Real}(G::InterpGrid{T,N}, x::AbstractVector{R}, xrest::AbstractVector{R}...)
    if length(xrest) != N-1
        error("Dimensionality mismatch")
    end
    nx = length(x)
    nrest = [length(y) for y in xrest]
    v = Array(T, nx, nrest...)
    for c in Counter(nrest)
        for i = 1:nx
            setx(G, x[i], ntuple(N-1, i->xrest[i][c[i]])...)  # FIXME performance?? May not matter...
            v[i,c...] = _getindex(G)
        end
    end
end


#### Non-uniform grid interpolation ####

# Currently supports only 1d, nearest-neighbor or linear
# Consequently, the internal representation may change in the future
# BCperiodic and BCreflect not supported
type InterpIrregular{T<:FloatingPoint, N, BC<:BoundaryCondition, IT<:InterpType} <: AbstractArray{T,N}
    grid::Vector{Vector{T}}
    coefs::Array{T,N}
    x::Vector{T}
    fillval::T  # used only for BCfill (if ever)
end
InterpIrregular{T<:FloatingPoint, BC<:BoundaryCondition, IT<:Union(InterpNearest,InterpLinear)}(grid::Vector{T}, A::AbstractVector, ::Type{BC}, ::Type{IT}) =
    InterpIrregular(Vector{T}[grid], A, BC, IT) # special 1d syntax
InterpIrregular{T<:FloatingPoint, BC<:BoundaryCondition, IT<:Union(InterpNearest,InterpLinear)}(grid::(Vector{T}...), A::AbstractVector, ::Type{BC}, ::Type{IT}) =
    InterpIrregular(Vector{T}[grid...], A, BC, IT)
function InterpIrregular{T<:FloatingPoint, BC<:BoundaryCondition, IT<:Union(InterpNearest,InterpLinear)}(grid::Vector{Vector{T}}, A::AbstractArray, ::Type{BC}, ::Type{IT})
    if length(grid) != 1
        error("Sorry, for now only 1d is supported")
    end
    if BC == BCreflect || BC == BCperiodic
        error("Sorry, reflecting or periodic boundary conditions not yet supported")
    end
    for i = 1:length(grid)
        if !issorted(grid[i])
            error("Coordinates must be supplied in increasing order")
        end
    end
    grid = copy(grid)
    coefs = convert(Array{T}, A)
    x = zeros(T, ndims(A))
    InterpIrregular{T, ndims(A), BC, IT}(grid, coefs, x, nan(T))
end
function InterpIrregular{T<:FloatingPoint, IT<:InterpType}(grid, A::Array{T}, f::Number, ::Type{IT})
    iu = InterpIrregular(grid, A, BCfill, IT)
    iu.fillval = f
    iu
end

function _getindexii{T,BC<:Union(BCfill,BCna,BCnan)}(G::InterpIrregular{T,1,BC}, x::Real)
    g = G.grid[1]
    i = (x == g[1]) ? 2 : searchsortedfirst(g, x)
    (i == 1 || i == length(g)+1) ? G.fillval : _interpu(x, g, i, G.coefs, interptype(G))
end
function _getindexii{T}(G::InterpIrregular{T,1,BCnil}, x::Real)
    g = G.grid[1]
    i = (x == g[1]) ? 2 : searchsortedfirst(g, x)
    (i == 1 || i == length(g)+1) ? error(BoundsError) : _interpu(x, g, i, interptype(G))
end
# This next is necessary for precedence
getindex(G::InterpIrregular, x::Real) = _getindexii(G, x)

_interpu(x, g, i, coefs, ::Type{InterpNearest}) = (x-g[i-1] < g[i]-x) ? coefs[i-1] : coefs[i]
function _interpu(x, g, i, coefs, ::Type{InterpLinear})
    f = (x-g[i-1])/(g[i]-g[i-1])
    (1-f)*coefs[i-1] + f*coefs[i]
end

# Low-level interpolation interface

# Low-level interpolation. The process is split into two parts:
#   1. Identify the neighbors that will contribute and calculate the
#      interpolation coefficients
#   2. Use the neighbor indices and coefficients to produce
#      interpolated values
# Splitting it this way helps support efficient multi-valued
# interpolation and gradient computation.
#
# set_position handles step 1 of this process.
function set_position{T,BC<:BoundaryCondition,IT<:InterpType}(ic::InterpGridCoefs{T,IT}, ::Type{BC}, calc_grad::Bool, x::Vector{T})
    N = ndims(ic)
    if length(x) != N
        error("Dimensionality mismatch")
    end
    valid = true
    for idim = 1:N
        if !isvalid(BC, IT, x[idim], ic.dims[idim])
            valid = false
            break
        end
    end
    ic.valid = valid
    wrap = false
    if valid
        ib::Int = 0
        for idim = 1:N
            ix::Int, dx::T, twrap::Bool = interp_coords_1d(ic.coord1d[idim], BC, IT, x[idim], ic.dims[idim])
            wrap = wrap | twrap
            ib += (ix-1)*ic.strides[idim]
            interp_coefs_1d(ic.coef1d[idim], BC, IT, dx)
            if calc_grad
                interp_gcoefs_1d(ic.gcoef1d[idim], BC, IT, dx)
            end
        end
        if wrap
            interp_index_coef(ic.index, ic.coef, ic.coord1d, ic.coef1d, ic.strides)
        else
            ic.offset_base = ib
            interp_coef(ic.coef, ic.coef1d)
        end
    end
    ic.wrap = wrap
end

# Once you're done calculating interpolated _values_, call this function
# for each component of the _gradient_.
# It effectively re-does step #1 above for the gradient component,
# although it skips over the calculation of the neighbor indices. You
# then call "interp" below to evaluate the gradient, just as you to
# evaluate the value. Put each component in a loop to get all the
# gradient components.
# Naturally, if you're interpolating a multi-valued function, it's far
# better to evaluate all values, then all first components of the
# gradient, then all second components of the gradient, etc, than it
# is to recalculate the coefficients for each of the multiple values.
function set_gradient_coordinate(ic::InterpGridCoefs, i::Int)
    if !ic.valid
        return
    end
    N = ndims(ic)
    if i < 1 || i > N
        error("Wrong dimension index")
    end
    for idim = 1:N
        if idim == i
            ic.c1d[idim] = ic.gcoef1d[idim]
        else
            ic.c1d[idim] = ic.coef1d[idim]
        end
    end
    interp_coef(ic.coef, ic.c1d)# , npoints(IT), N)
end

# "interp" evaluates the interpolation. Call "set_position" first. If
# you want to also evaluate the gradient, call
# "set_gradient_coordinate" followed by "interp" for each component.
function interp(ic::InterpGridCoefs, A::AbstractArray, index::Int)
    if !ic.valid
        return nan(eltype(A))
    end
    coef = ic.coef
    if ic.wrap
        offset = ic.index
        index -= 1
    else
        offset = ic.offset
        index += ic.offset_base
    end
    val = coef[1]*A[offset[1]+index]
    for i = 2:length(coef)
        val += coef[i]*A[offset[i]+index]
    end
    return convert(eltype(A), val)
end
interp(ic::InterpGridCoefs, A::AbstractArray) = interp(ic, A, 1)

# Change the array dimensions
function set_size(ic::InterpGridCoefs, dims, strides)
    N = ndims(ic)
    if length(strides) != N
        error("Strides do not have the correct dimensionality")
    end
    if length(dims) != N
        error("Dimensions do not have the correct dimensionality")
    end
    d = [dims...]
    s = [strides...]
    for idim = 1:N
        ic.dims[idim] = d[idim]
        ic.strides[idim] = s[idim]
        interp_coords_1d(ic.coord1d[idim], interptype(ic))
    end
    interp_index(ic.offset, ic.coord1d, s)
end
function set_size(ic::InterpGridCoefs, dims)
    N = length(dims)
    s = Array(Int, N)
    s[1] = 1
    for i = 1:N-1
        s[i+1] = s[i]*dims[i]
    end
    set_size(ic, dims, s)
end

# strides is supplied separately to allow working with subarrays,
# and/or specific dimensions. For example, an RGB image might be
# interpolated with respect to the spatial dimensions but not the
# dimension that holds color.
function InterpGridCoefs{T<:FloatingPoint,IT<:InterpType}(::Type{T}, ::Type{IT}, dims::Union(Dims,Vector{Int}), strides::Union(Dims,Vector{Int}))
    N = length(strides)
    if length(dims) != N
        error("Length of dims and strides must match")
    end
    coord1d = Array(Vector{Int}, N)
    coef1d = Array(Vector{T}, N)
    gcoef1d = Array(Vector{T}, N)
    c1d = Array(Vector{T}, N)
    l = npoints(IT)
    for idim = 1:N
        coord1d[idim] = Array(Int, l)
        coef1d[idim] = Array(T, l)
        gcoef1d[idim] = Array(T, l)
        # do not allocate entries for c1d
    end
    n_coef = l^N
    offset = Array(Int, n_coef)
    index = Array(Int, n_coef)
    coef = Array(T, n_coef)
    # Pre-calculate the default offset from the strides
    for idim = 1:N
        interp_coords_1d(coord1d[idim], IT)
    end
    interp_index(offset, coord1d, strides)
    InterpGridCoefs{T,IT}(coord1d,coef1d,gcoef1d,c1d,[dims...],[strides...],offset,0,index,false,false,coef)
end
InterpGridCoefs{IT<:InterpType}(A::Array, ::Type{IT}) = InterpGridCoefs(eltype(A), IT, [size(A)...], [strides(A)...])



# Interpolation support routines

npoints(::Type{InterpNearest}) = 1
npoints(::Type{InterpLinear}) = 2
npoints(::Type{InterpQuadratic}) = 3

# Test whether a given coordinate will yield an interpolated result
isvalid{BC<:BoundaryCondition,IT<:InterpType}(::Type{BC},::Type{IT}, x, len::Int) = true
isvalid{BC<:Union(BCnil,BCnan,BCna)}(::Type{BC}, ::Type{InterpNearest}, x, len::Int) = x >= 0.5 && x-0.5 <= len
isvalid{BC<:Union(BCnil,BCnan,BCna)}(::Type{BC}, ::Type{InterpLinear}, x, len::Int) = x >= 1 && x <= len
isvalid{BC<:Union(BCnil,BCnan,BCna)}(::Type{BC}, ::Type{InterpQuadratic}, x, len::Int) = x >= 1 && x <= len


# version for offsets
function interp_coords_1d(coord1d::Vector{Int}, ::Type{InterpNearest})
    coord1d[1] = 0
end
# version for indices
function interp_coords_1d{BC<:BoundaryCondition}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpNearest}, x, len::Int)
    ix = wrap(BC, iround(x), len)
    coord1d[1] = ix
    return ix, 0, false
end

# version for offsets
function interp_coords_1d(coord1d::Vector{Int}, ::Type{InterpLinear})
    coord1d[1] = 0
    coord1d[2] = 1
end
# version for indices
function interp_coords_1d{T,BC<:BoundaryCondition}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpLinear}, x::T, len::Int)
    ix::Int = wrap(BC, ifloor(x), len)
    dx::T = x-trunc(x)
    coord1d[1] = ix
    iswrap::Bool = (ix == len && dx > 0)
    coord1d[2] = wrap(BC, ix+1, len)
    return ix, dx, iswrap
end
function interp_coords_1d{T}(coord1d::Vector{Int}, ::Type{BCreflect}, ::Type{InterpLinear}, x::T, len::Int)
    ix = mod(ifloor(x)-1, 2*len)
    dx = x-trunc(x)
    if ix < len
        ix += 1
        ixp = ix+1
        if ixp > len
            ixp = len
        end
    else
        ix = 2*len-ix-1
        dx = one(T)-dx
        ixp = ix+1
        if ix == 0
            ix = 1
        end
    end
    coord1d[1] = ix
    coord1d[2] = wrap(BCreflect, ixp, len)
    iswrap = (ix == ixp && dx > 0)
    return ix, dx, iswrap
end

# version for offsets
function interp_coords_1d(coord1d::Vector{Int}, ::Type{InterpQuadratic})
    coord1d[1] = -1
    coord1d[2] = 0
    coord1d[3] = 1
end
# versions for indices
function interp_coords_1d{BC<:BoundaryCondition}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpQuadratic}, x, len::Int)
    ix = iround(x)
    dx = x-ix
    ix = wrap(BC, ix, len)
    coord1d[2] = ix
    iswrap = (ix == 1 || ix == len)
    if iswrap
        coord1d[1] = wrap(BC, ix-1, len)
        coord1d[3] = wrap(BC, ix+1, len)
    else
        coord1d[1] = ix-1
        coord1d[3] = ix+1
    end
    return ix, dx, iswrap
end
# for InterpQuadratic, several require special handling
# BCnil, Bnan, BCna: for 1 <= x <= 1.5, continue the quadratic centered at x=2
function interp_coords_1d{BC<:Union(BCnil,BCnan,BCna)}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpQuadratic}, x, len::Int)
    if x > 1.5 && x+0.5 < len
        ix = iround(x)
    elseif x <= 1.5
        ix = 2
    else
        ix = len-1
    end
    dx = x-ix
    coord1d[1] = ix-1
    coord1d[2] = ix
    coord1d[3] = ix+1
    return ix, dx, false
end
# BCnearest & BCfill: for 1 <= x <= 1.5, ensure the slope tends to 0 at x=1
function interp_coords_1d{T,BC<:Union(BCnearest,BCfill)}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpQuadratic}, x::T, len::Int)
    if x > 1.5 && x+0.5 < len
        ix = iround(x)
        coord1d[1] = ix-1
        coord1d[2] = ix
        coord1d[3] = ix+1
        iswrap = false
        dx = x-ix
    elseif x <= 1.5
        ix = one(T)
        coord1d[1] = 2
        coord1d[2] = 1
        coord1d[3] = 2
        iswrap = true
        if x < 1
            dx = zero(T)
        else
            dx = x-ix
        end
    else
        ix = convert(T, len)
        coord1d[1] = len-1
        coord1d[2] = len
        coord1d[3] = len-1
        iswrap = true
        if x > len
            dx = zero(T)
        else
            dx = x-ix
        end
    end
    return ix, dx, iswrap
end
function interp_coords_1d(coord1d::Vector{Int}, ::Type{BCreflect}, ::Type{InterpQuadratic}, x, len::Int)
    ix = iround(x)
    dx = x-ix
    ix = mod(ix-1, 2*len)
    if ix < len
        ix += 1
    else
        dx = -dx
        ix = 2*len-ix
    end
    coord1d[2] = ix
    iswrap = (ix == 1 || ix == len)
    if iswrap
        coord1d[1] = wrap(BCreflect, ix-1, len)
        coord1d[3] = wrap(BCreflect, ix+1, len)
    else
        coord1d[1] = ix-1
        coord1d[3] = ix+1
    end
    return ix, dx, iswrap
end

function interp_coefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpNearest}, dx::T)
    coef1d[1] = one(T)
end
function interp_gcoefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpQuadratic}, dx::T)
    coef1d[1] = zero(T)
end

function interp_coefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpLinear}, dx::T)
    coef1d[1] = 1.0-dx
    coef1d[2] = dx
end
function interp_gcoefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpQuadratic}, dx::T)
    coef1d[1] = -one(T)
    coef1d[2] = one(T)
end

function interp_coefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpQuadratic}, dx::T)
    coef1d[1] = (dx-0.5)^2/2
    coef1d[2] = 0.75-dx^2
    coef1d[3] = (dx+0.5)^2/2
end
function interp_gcoefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpQuadratic}, dx::T)
    coef1d[1] = dx-0.5
    coef1d[2] = -2dx
    coef1d[3] = dx+0.5
end

eltype{T, N, BC, IT}(G::InterpGrid{T, N, BC, IT}) = T
ndims{T, N, BC, IT}(G::InterpGrid{T, N, BC, IT}) = N
boundarycondition{T, N, BC, IT}(G::InterpGrid{T, N, BC, IT}) = BC
interptype{T, N, BC, IT}(G::InterpGrid{T, N, BC, IT}) = IT
size(G::InterpGrid) = size(G.coefs)
size(G::InterpGrid, i::Integer) = size(G.coefs, i)

eltype{T, N, BC, IT}(G::InterpIrregular{T, N, BC, IT}) = T
ndims{T, N, BC, IT}(G::InterpIrregular{T, N, BC, IT}) = N
boundarycondition{T, N, BC, IT}(G::InterpIrregular{T, N, BC, IT}) = BC
interptype{T, N, BC, IT}(G::InterpIrregular{T, N, BC, IT}) = IT
size(G::InterpIrregular) = size(G.coefs)
size(G::InterpIrregular, i::Integer) = size(G.coefs, i)

eltype{T,IT<:InterpType}(ic::InterpGridCoefs{T,IT}) = T
interptype{IT<:InterpType,T}(ic::InterpGridCoefs{T,IT}) = IT
ndims(ic::InterpGridCoefs) = length(ic.coord1d)
show(io::IO, ic::InterpGridCoefs) = print(io, "InterpGridCoefs{", eltype(ic), ",", interptype(ic), "}")


# Generalized interpolation of higher order than InterpLinear requires
# inversion of the interpolation operator. See, for example, the
# Thevanez citation above.
const Q3inv = [7/8 1/4 -1/8; -1/8 5/4 -1/8; -1/8 1/4 7/8] # for handling "snippets" of size 3 along each dimension (InterpQuadratic)
# This works in place. If instead it allocated the output for you, then
# calling multiple times (e.g., to apply inversions for different
# interptypes along different dimensions) would result in unnecessary
# allocations.
interp_invert!{BC<:BoundaryCondition}(A::Array, ::Type{BC}, ::Type{InterpNearest}, dimlist) = A
interp_invert!{BC<:BoundaryCondition}(A::Array, ::Type{BC}, ::Type{InterpLinear}, dimlist) = A
function interp_invert!{BC<:BoundaryCondition}(A::Array, ::Type{BC}, ::Type{InterpQuadratic}, dimlist)
    sizeA = [size(A)...]
    stridesA = [strides(A)...]
    for idim = dimlist
        n = size(A, idim)
        # Set up the tridiagonal system
        du = fill(convert(eltype(A), 1/8), n-1)
        d = fill(convert(eltype(A), 3/4), n)
        dl = copy(du)
        M = _interp_invert_matrix(BC, InterpQuadratic, dl, d, du)
        sizeA[idim] = 1  # don't iterate over the dimension we're solving on
        for cc in Counter(sizeA)
            rng = Range(sum((cc-1).*stridesA) + 1, stridesA[idim], n)
            solve(A, rng, M, A, rng) # in-place
        end
        sizeA[idim] = size(A, idim)
    end
end
interp_invert!{BC<:Union(BoundaryCondition,Number)}(A::Array, ::Type{BC}, IT) = interp_invert!(A, BC, IT, 1:ndims(A))
interp_invert!{BC<:Union(BoundaryCondition,Number)}(A::Array, ::Type{BC}, IT, dimlist...) = interp_invert!(A, BC, IT, dimlist)

function _interp_invert_matrix{BC<:Union(BCnil,BCnan,BCna),T}(::Type{BC}, ::Type{InterpQuadratic}, dl::Vector{T}, d::Vector{T}, du::Vector{T})
    # For these, the quadratic centered on x=2 is continued down to
    # 1 rather than terminating at 1.5
    n = length(d)
    d[1] = d[n] = 9/8
    dl[n-1] = du[1] = -1/4
    MT = Tridiagonal(dl, d, du)
    # Woodbury correction to add 1/8 for row 1, col 3 and row n, col n-2
    U = zeros(T, n, 2)
    V = zeros(T, 2, n)
    C = zeros(T, 2, 2)
    C[1,1] = C[2,2] = 1/8
    U[1,1] = U[n,2] = 1
    V[1,3] = V[2,n-2] = 1
    M = Woodbury(MT, U, C, V)
end
function _interp_invert_matrix{T}(::Type{BCreflect}, ::Type{InterpQuadratic}, dl::Vector{T}, d::Vector{T}, du::Vector{T})
    n = length(d)
    d[1] += 1/8
    d[n] += 1/8
    M = Tridiagonal(dl, d, du)
end
function _interp_invert_matrix{T}(::Type{BCperiodic}, ::Type{InterpQuadratic}, dl::Vector{T}, d::Vector{T}, du::Vector{T})
    n = length(d)
    MT = Tridiagonal(dl, d, du)
    # Woodbury correction to wrap around
    U = zeros(T, n, 2)
    V = zeros(T, 2, n)
    C = zeros(T, 2, 2)
    C[1,1] = C[2,2] = 1/8
    U[1,1] = U[n,2] = 1
    V[1,n] = V[2,1] = 1
    M = Woodbury(MT, U, C, V)
end
function _interp_invert_matrix{T,BC<:Union(BCnearest,BCfill)}(::Type{BC}, ::Type{InterpQuadratic}, dl::Vector{T}, d::Vector{T}, du::Vector{T})
    n = length(d)
    du[1] += 1/8
    dl[n-1] += 1/8
    M = Tridiagonal(dl, d, du)
end

# We have 3 functions for computing indices and coefficients:
#   interp_coef: fast, computes just the coefficients
#   interp_index_coef: fast, computes both indices and coefficients
#   interp_index: slow, computes just the indices
# The latter is rarely used (presumably, only upon InterpGridCoefs
# allocation), and hence should not be performance-critical, so the
# code is kept simple.
# The motivation for this organization is that in the "generic" case
# (see below), the overhead of looping is such that it's better to
# compute the indices and coefficients at the same time. However, when
# you don't need the indices, for fixed dimensions it's a significant
# savings (roughly 2x, at least for 3d) to just compute the
# coefficients.
function interp_index(index::Vector{Int}, coord1d::Vector{Vector{Int}}, s)
    N = length(coord1d)
    l = map(length, coord1d)
    i = 1
    for c in Counter(l)
        indx = s[1]*coord1d[1][c[1]]
        for idim = 2:N
            indx += s[idim]*coord1d[idim][c[idim]]
        end
        index[i] = indx
        i += 1
    end
end
# This is the main bottleneck, so it pays handsomely to specialize on
# dimension.
function interp_index_coef{T}(indices::Array{Int}, coefs::Array{T}, coord1d::Vector{Vector{Int}}, coef1d::Vector{Vector{T}}, strides)
    n_dims = length(coef1d)
    l = length(coef1d[1])
    if n_dims == 1
        copy!(coefs, coef1d[1])
        s = strides[1]
        c = coord1d[1]
        for i = 1:l
            indices[i] = (c[i]-1)*s + 1
        end
    elseif n_dims == 2
        s1 = strides[1]
        s2 = strides[2]
        cd1 = coord1d[1]
        cd2 = coord1d[2]
        cf1 = coef1d[1]
        cf2 = coef1d[2]
        ind = 1
        for i2 = 1:l
            offset = (cd2[i2]-1)*s2
            p = cf2[i2]
            for i1 = 1:l
                indices[ind] = offset + (cd1[i1]-1)*s1 + 1
                coefs[ind] = p*cf1[i1]
                ind += 1
            end
        end
    elseif n_dims == 3
        s1 = strides[1]
        s2 = strides[2]
        s3 = strides[3]
        cd1 = coord1d[1]
        cd2 = coord1d[2]
        cd3 = coord1d[3]
        cf1 = coef1d[1]
        cf2 = coef1d[2]
        cf3 = coef1d[3]
        ind = 1
        for i3 = 1:l
            offset3 = (cd3[i3]-1)*s3
            p3 = cf3[i3]
            for i2 = 1:l
                offset2 = offset3 + (cd2[i2]-1)*s2
                p2 = p3*cf2[i2]
                for i1 = 1:l
                    indices[ind] = offset2 + (cd1[i1]-1)*s1 + 1
                    coefs[ind] = p2*cf1[i1]
                    ind += 1
                end
            end
        end
    elseif n_dims == 4
        s1 = strides[1]
        s2 = strides[2]
        s3 = strides[3]
        s4 = strides[4]
        cd1 = coord1d[1]
        cd2 = coord1d[2]
        cd3 = coord1d[3]
        cd4 = coord1d[4]
        cf1 = coef1d[1]
        cf2 = coef1d[2]
        cf3 = coef1d[3]
        cf4 = coef1d[4]
        ind = 1
        for i4 = 1:l
            offset4 = (cd4[i4]-1)*s4
            p4 = cf4[i4]
            for i3 = 1:l
                offset3 = offset4 + (cd3[i3]-1)*s3
                p3 = p4*cf3[i3]
                for i2 = 1:l
                    offset2 = offset3 + (cd2[i2]-1)*s2
                    p2 = p3*cf2[i2]
                    for i1 = 1:l
                        indices[ind] = offset2 + (cd1[i1]-1)*s1 + 1
                        coefs[ind] = p2*cf1[i1]
                        ind += 1
                    end
                end
            end
        end
    else
        interp_index_coef_generic(indices, coefs, coord1d, coef1d, strides)
    end
end
function interp_coef{T}(coefs::Array{T}, coef1d::Vector{Vector{T}})
    n_dims = length(coef1d)
    l = length(coef1d[1])
    if n_dims == 1
        copy!(coefs, coef1d[1])
    elseif n_dims == 2
        cf1 = coef1d[1]
        cf2 = coef1d[2]
        ind = 1
        for i2 = 1:l
            p = cf2[i2]
            for i1 = 1:l
                coefs[ind] = p*cf1[i1]
                ind += 1
            end
        end
    elseif n_dims == 3
        cf1 = coef1d[1]
        cf2 = coef1d[2]
        cf3 = coef1d[3]
        ind = 1
        for i3 = 1:l
            p3 = cf3[i3]
            for i2 = 1:l
                p2 = p3*cf2[i2]
                for i1 = 1:l
                    coefs[ind] = p2*cf1[i1]
                    ind += 1
                end
            end
        end
    elseif n_dims == 4
        cf1 = coef1d[1]
        cf2 = coef1d[2]
        cf3 = coef1d[3]
        cf4 = coef1d[4]
        ind = 1
        for i4 = 1:l
            p4 = cf4[i4]
            for i3 = 1:l
                p3 = p4*cf3[i3]
                for i2 = 1:l
                    p2 = p3*cf2[i2]
                    for i1 = 1:l
                        coefs[ind] = p2*cf1[i1]
                        ind += 1
                    end
                end
            end
        end
    else
        interp_coef_generic(coefs, coef1d)
    end
end
# Compared to the dimension-specialized versions, these next two will
# be slow due to having many more arrayref operations. But at least it
# avoids an extra loop compared to the naive implementation using
# Counter, so it could be a lot worse.
# Consider making something like make_arrayind_loop_nest?
function interp_index_coef_generic{T}(indices::Array{Int}, coefs::Array{T}, coord1d::Vector{Vector{Int}}, coef1d::Vector{Vector{T}}, strides)
    n_dims = length(coord1d)
    l = length(coord1d[1])
    c = fill(1, n_dims)
    offset = Array(Int, n_dims)
    p = Array(T, n_dims)
    offset[end] = (coord1d[end][1]-1)*strides[end]
    p[end] = coef1d[end][1]
    for idim = n_dims-1:-1:1
        offset[idim] = offset[idim+1] + (coord1d[idim][1]-1)*strides[idim]
        p[idim] = p[idim+1] * coef1d[idim][1]
    end
    ind = 1
    while c[n_dims] <= l
        indices[ind] = offset[1]+1
        coefs[ind] = p[1]
        ind += 1
        c[1] += 1
        thisc = c[1]
        if thisc <= l
            if n_dims > 1
                p[1] = p[2] * coef1d[1][thisc]
                offset[1] = offset[2] + (coord1d[1][thisc]-1)*strides[1]
            else
                p[1] = coef1d[1][thisc]
                offset[1] = (coord1d[1][thisc]-1)*strides[1]
            end
        else
            idim = 1
            while c[idim] > l && idim < n_dims
                c[idim] = 1
                c[idim+1] += 1
                idim += 1
            end
            if idim == n_dims
                thisc = c[idim]
                if thisc <= l
                    p[idim] = coef1d[idim][thisc]
                    offset[idim] = (coord1d[idim][thisc]-1)*strides[idim]
                    idim -= 1
                end
            end
            while idim > 0 && idim < n_dims
                thisc = c[idim]
                offset[idim] = offset[idim+1] + (coord1d[idim][thisc]-1)*strides[idim]
                p[idim] = p[idim+1] * coef1d[idim][thisc]
                idim -= 1
            end
        end
    end
end
function interp_coef_generic{T}(indices::Array{Int}, coefs::Array{T}, coord1d::Vector{Vector{Int}}, coef1d::Vector{Vector{T}}, strides)
    n_dims = length(coord1d)
    l = length(coord1d[1])
    c = ones(Int, n_dims)
    offset = Array(Int, n_dims)
    p = Array(T, n_dims)
    p[end] = coef1d[end][1]
    for idim = n_dims-1:-1:1
        p[idim] = p[idim+1] * coef1d[idim][1]
    end
    ind = 1
    while c[n_dims] <= l
        coefs[ind] = p[1]
        ind += 1
        c[1] += 1
        thisc = c[1]
        if thisc <= l
            if n_dims > 1
                p[1] = p[2] * coef1d[1][thisc]
            else
                p[1] = coef1d[1][thisc]
            end
        else
            idim = 1
            while c[idim] > l && idim < n_dims
                c[idim] = 1
                c[idim+1] += 1
                idim += 1
            end
            if idim == n_dims
                thisc = c[idim]
                if thisc <= l
                    p[idim] = coef1d[idim][thisc]
                    idim -= 1
                end
            end
            while idim > 0 && idim < n_dims
                thisc = c[idim]
                p[idim] = p[idim+1] * coef1d[idim][thisc]
                idim -= 1
            end
        end
    end
end

# Padding an array by 1 element on each boundary of chosen dimensions
# This is needed to support BCfill
function pad1_index(szt::Tuple, dimpad)
    szv = [szt...]
    for i in dimpad
        szv[i] += 2
    end
    N = length(szt)
    ind = cell(N)
    for i in 1:N
        if szv[i] > szt[i]
            ind[i] = 2:szv[i]-1
        else
            ind[i] = 1:szv[i]
        end
    end
    return szv, ind
end
pad1{BC<:BoundaryCondition}(Ain::AbstractArray, ::Type{BC}, dimpad...) = copy(Ain)  # no padding needed for most types
pad1(Ain::AbstractArray, ::Type{BCfill}, dimpad...) = error("For BCfill, specify the fill value instead of the type BCfill")
function pad1(Ain::AbstractArray, f::Number, dimpad...)
    sz, ind = pad1_index(size(Ain), dimpad)
    A = fill(convert(eltype(Ain), f), sz...)
    A[ind...] = Ain
    return A
end


#### Restrict/prolong operators ####

# These are operators for solving PDEs via multigrid. These are also
# very fast routines for changing the size of an array by twofold,
# using linear interpolation for enlarging, and properly anti-aliasing
# for shrinking.

typealias LapackScalar Union(Float32, Float64, Complex64, Complex128)
# prolong for BLAS types
function prolong{T<:LapackScalar}(A::Array{T}, dim::Integer, len::Integer)
    sz = [size(A)...]
    sz[dim] = prolong_size(sz[dim], len)
    P = zeros(eltype(A), to_tuple(sz))
    sA = [strides(A)...]
    skipA = sA[dim]
    sP = [strides(P)...]
    skipP = sP[dim]
    n = size(A, dim)
    sz[dim] = 1
    if isodd(len)
        # Copy on-grid, and interpolate at half-grid points
        for indices = Counter(sz)
            startA = sum((indices-1).*sA)+1
            startP = sum((indices-1).*sP)+1
            # handle the on-grid points
            copy!(P, Range(startP, 2*skipP, n), A, Range(startA, skipA, n))
            # handle the off-grid points (linear interpolation)
            rP = Range(startP+skipP, 2*skipP, n-1)
            Base.LinAlg.BLAS.axpy!(0.5, A, Range(startA, skipA, n-1), P, rP)
            Base.LinAlg.BLAS.axpy!(0.5, A, Range(startA+skipA, skipA, n-1), P, rP)
        end
    else
        # Interpolate at 1/4 and 3/4 points
        for indices = Counter(sz)
            startA = sum((indices-1).*sA)+1
            startP = sum((indices-1).*sP)+1
            rA1 = Range(startA, skipA, n-1)
            rA2 = Range(startA+skipA, skipA, n-1)
            rP1 = Range(startP, 2*skipP, n-1)
            rP2 = Range(startP+skipP, 2*skipP, n-1)
            Base.LinAlg.BLAS.axpy!(0.75, A, rA1, P, rP1)
            Base.LinAlg.BLAS.axpy!(0.25, A, rA2, P, rP1)
            Base.LinAlg.BLAS.axpy!(0.25, A, rA1, P, rP2)
            Base.LinAlg.BLAS.axpy!(0.75, A, rA2, P, rP2)
        end
    end
    return P
end
# prolong for non-BLAS types (including tiled arrays)
function prolong{T}(A::Array{T}, dim::Integer, len::Integer)
    sz = [size(A)...]
    sz[dim] = prolong_size(sz[dim], len)
    P = Array(T, to_tuple(sz))
    sA = [strides(A)...]
    skipA = sA[dim]
    sP = [strides(P)...]
    skipP = sP[dim]
    n = size(A, dim)
    sz[dim] = 1   # prolonged dimension is handled directly
    if isodd(len)
        for indices = Counter(sz)
            iA = sum((indices-1).*sA)+1
            iP = sum((indices-1).*sP)+1
            for i = 1:n-1
                P[iP] = A[iA]
                P[iP+skipP] = (A[iA] + A[iA+skipA])/2
                iA += skipA
                iP += 2*skipP
            end
            P[iP] = A[iA]
        end
    else
        for indices = Counter(sz)
            iA = sum((indices-1).*sA)+1
            iP = sum((indices-1).*sP)+1
            for i = 1:n-1
                P[iP] = (3*A[iA] + A[iA+skipA])/4
                P[iP+skipP] = (A[iA] + 3*A[iA+skipA])/4
                iA += skipA
                iP += 2*skipP
            end
        end
    end
    return P
end

function prolong(A::Array, sz::Array{Int})
    if ndims(A) != size(sz,1)
        error("Array of sizes must have as many rows as dimensions of A")
    end
    Ap = A
    for j = 1:size(sz, 2)
        for i = 1:size(sz, 1)
            if sz[i,j] > size(Ap, i)
                Ap = prolong(Ap, i, sz[i,j])
            end
        end
    end
    return Ap
end

# restrict for BLAS types
function restrict{T<:LapackScalar}(A::Array{T}, dim::Integer, scale::Real)
    sz = [size(A)...]
    sz[dim] = restrict_size(sz[dim])
    R = zeros(eltype(A), to_tuple(sz))
    sA = [strides(A)...]
    skipA = sA[dim]
    sR = [strides(R)...]
    skipR = sR[dim]
    n = sz[dim]
    sz[dim] = 1
    if isodd(size(A, dim))
        for indices = Counter(sz)
            startA = sum((indices-1).*sA)+1
            startR = sum((indices-1).*sR)+1
            Base.LinAlg.BLAS.axpy!(scale, A, Range(startA, 2*skipA, n), R, Range(startR, skipR, n))
            rA = Range(startA+skipA, 2*skipA, n-1)
            rR = Range(startR, skipR, n-1)
            Base.LinAlg.BLAS.axpy!(scale/2, A, rA, R, rR)
            rR = Range(startR+skipR, skipR, n-1)
            Base.LinAlg.BLAS.axpy!(scale/2, A, rA, R, rR)
        end
    else
        for indices = Counter(sz)
            startA = sum((indices-1).*sA)+1
            startR = sum((indices-1).*sR)+1
            rA = Range(startA, 2*skipA, n-1)
            rR = Range(startR, skipR, n-1)
            Base.LinAlg.BLAS.axpy!(0.75*scale, A, rA, R, rR)
            rA = Range(startA+skipA, 2*skipA, n-1)
            Base.LinAlg.BLAS.axpy!(0.25*scale, A, rA, R, rR)
            rR = Range(startR+skipR, skipR, n-1)
            Base.LinAlg.BLAS.axpy!(0.75*scale, A, rA, R, rR)
            rA = Range(startA, 2*skipA, n-1)
            Base.LinAlg.BLAS.axpy!(0.25*scale, A, rA, R, rR)
        end
    end
    return R
end
# restrict for non-BLAS types
function restrict{T}(A::Array{T}, dim::Integer, scale::Real)
    sz = [size(A)...]
    sz[dim] = restrict_size(sz[dim])
    R = Array(eltype(A), to_tuple(sz))
    sA = [strides(A)...]
    skipA = sA[dim]
    sR = [strides(R)...]
    skipR = sR[dim]
    n = sz[dim]
    sz[dim] = 1
    if isodd(size(A, dim))
        for indices = Counter(sz)
            iA = sum((indices-1).*sA)+1
            iR = sum((indices-1).*sR)+1
            R[iR] = scale*(A[iA] + 0.5*A[iA+skipA])
            for i = 2:n-1
                iA += 2*skipA
                iR += skipR
                R[iR] = scale*(0.5*A[iA-skipA] + A[iA] + 0.5*A[iA+skipA])
            end
            R[iR+skipR] = scale*(0.5*A[iA+skipA] + A[iA+2*skipA])
        end
    else
        for indices = Counter(sz)
            iA = sum((indices-1).*sA)+1
            iR = sum((indices-1).*sR)+1
            R[iR] = (0.75*scale)*A[iA] + (0.25*scale)*A[iA+skipA]
            iR += skipR
            for i = 2:n-1
                R[iR] = (0.25*scale)*A[iA] + (0.75*scale)*A[iA+skipA] + (0.75*scale)*A[iA+2*skipA] + (0.25*scale)*A[iA+3*skipA]
                iR += skipR
                iA += 2*skipA
            end
            R[iR] = (0.25*scale)*A[iA] + (0.75*scale)*A[iA+skipA]
        end
    end
    return R
end
restrict{T}(A::Array{T}, dim::Integer) = restrict(A, dim, one(T))
function restrict(A::Array, flag::Union(Array{Bool},BitArray), scale::Real)
    func = (A, dim) -> restrict(A, dim, scale)
    return mapdim(func, A, flag)
end
restrict{T}(A::Array{T}, flag::Union(Array{Bool},BitArray)) = restrict(A, flag, one(eltype(T)))

# "Balanced" restriction and prolongation

# If you take an array of ones, restrict it, and then prolong it
# (i.e., compute P*R*A, where P is the prolongation operator and R is
# the restriction operator), it smooths the array. However, by
# comparison to all other points in the array the edges will tend
# towards zero. The "balanced" versions of restrict/prolong ensure
# that each point of the array is used equally, i.e., that sum(P*R, 1)
# is constant, and hence the edges will be preserved.
function restrictb{T}(A::Array{T}, dim::Integer, scale::Real)
    a::T
    b::T
    Ar = restrict(A, dim, scale)
    sz = [size(Ar)...]
    sz[dim] = 1
    s = [strides(A)...]
    sr = [strides(Ar)...]
    offset_end = (size(A,dim)-1)*s[dim]
    offset_endr = (size(Ar,dim)-1)*sr[dim]
    Astep = s[dim]
    if isodd(size(A,dim))
        b = scale/sqrt(3)
        a = 2*b
        for indices = Counter(sz)
            index = sum((indices-1).*s)+1
            indexr = sum((indices-1).*sr)+1
            Ar[indexr] = a*A[index] + b*A[index+Astep]
            Ar[indexr+offset_endr] = a*A[index+offset_end] + b*A[index+offset_end-Astep]
        end
    else
        b = scale/(2*sqrt(2))
        a = 3*b
        for indices = Counter(sz)
            index = sum((indices-1).*s)+1
            indexr = sum((indices-1).*sr)+1
            Ar[indexr] = a*A[index] + b*A[index+Astep]
            Ar[indexr+offset_endr] = a*A[index+offset_end] + b*A[index+offset_end-Astep]
        end
    end
    return Ar
end
restrictb{T}(A::Array{T}, dim::Integer) = restrictb(A, dim, one(T))
function restrictb(A::Array, flag::Union(Array{Bool},BitArray), scale::Real)
    func = (A, dim) -> restrictb(A, dim, scale)
    return mapdim(func, A, flag)
end
restrictb{T}(A::Array{T}, flag::Union(Array{Bool},BitArray)) = restrictb(A, flag, one(eltype(T)))
function prolongb{T}(A::Array{T}, dim::Integer, len::Integer)
    a::T
    b::T
    Ap = prolong(A, dim, len)
    sz = [size(Ap)...]
    sz[dim] = 1
    s = [strides(A)...]
    sp = [strides(Ap)...]
    offset_end = (size(A,dim)-1)*s[dim]
    offset_endp = (size(Ap,dim)-1)*sp[dim]
    Astep = s[dim]
    Apstep = sp[dim]
    if isodd(len)
        b = 1/sqrt(3)
        a = 2*b
        for indices = Counter(sz)
            index = sum((indices-1).*s)+1
            indexp = sum((indices-1).*sp)+1
            Ap[indexp] = a*A[index]
            Ap[indexp+Apstep] = b*A[index] + A[index+Astep]/2
            Ap[indexp+offset_endp] = a*A[index+offset_end]
            Ap[indexp+offset_endp-Apstep] = b*A[index+offset_end] + A[index+offset_end-Astep]/2
        end
    else
        b = 1/(2*sqrt(2))
        a = 3*b
        for indices = Counter(sz)
            index = sum((indices-1).*s)+1
            indexp = sum((indices-1).*sp)+1
            Ap[indexp] = a*A[index] + A[index+Astep]/4
            Ap[indexp+Apstep] = b*A[index] + (3/4)*A[index+Astep]
            Ap[indexp+offset_endp] = a*A[index+offset_end] + A[index+offset_end-Astep]/4
            Ap[indexp+offset_endp-Apstep] = b*A[index+offset_end] + (3/4)*A[index+offset_end-Astep]
        end
    end
    return Ap
end
function prolongb(A::Array, sz::Array{Int})
    if ndims(A) != size(sz,1)
        error("Array of sizes must have as many rows as dimensions of A")
    end
    Ap = A
    for j = 1:size(sz, 2)
        for i = 1:size(sz, 1)
            if sz[i,j] > size(Ap, i)
                Ap = prolongb(Ap, i, sz[i,j])
            end
        end
    end
    return Ap
end

# restrict, then "repair" the edges by linear extrapolation
# Do _not_ use this if you're doing gradient restriction/prolongation,
# as this breaks the chain rule. Just use ordinary restrict/prolong or
# their balanced "b" variants.
function restrict_extrap(A::Array, dim::Integer, scale::Real)
    Ar = restrict(A, dim, scale)
    sz = [size(Ar)...]
    sz[dim] = 1
    s = [strides(A)...]
    sr = [strides(Ar)...]
    offset_end = (size(A,dim)-1)*s[dim]
    offset_endr = (size(Ar,dim)-1)*sr[dim]
    if isodd(size(A,dim))
        for indices = Counter(sz)
            index = sum((indices-1).*s)+1
            indexr = sum((indices-1).*sr)+1
            Ar[indexr] = (2*scale)*A[index]
            Ar[indexr+offset_endr] = (2*scale)*A[index+offset_end]
        end
    else
        for indices = Counter(sz)
            index = sum((indices-1).*s)+1
            indexr = sum((indices-1).*sr)+1
            Ar[indexr] = (3*scale)*A[index] - scale*A[index+s[dim]]
            Ar[indexr+offset_endr] = (3*scale)*A[index+offset_end] - scale*A[index+offset_end-s[dim]]
        end
    end
    return Ar
end
restrict_extrap{T}(A::Array{T}, dim::Integer) = restrict_extrap(A, dim, one(T))
function restrict_extrap(A::Array, flag::Union(Array{Bool},BitArray), scale::Real)
    func = (A, dim) -> restrict_extrap(A, dim, scale)
    return mapdim(func, A, flag)
end
restrict_extrap{T}(A::Array{T}, flag::Union(Array{Bool},BitArray)) = restrict_extrap(A, flag, one(T))

function prolong_size(sz::Int, len::Int)
    if isodd(len)
        newsz = 2*sz-1
    else
        newsz = 2*(sz-1)
    end
    if newsz != len
        error("Cannot prolong a dimension of length ", sz, " to ", len)
    end
    return newsz
end

restrict_size(len::Int) = isodd(len) ? div(len+1,2) : div(len,2)+1
function restrict_size(szin::Union(Dims, Vector{Int}), flag::Union(Array{Bool},BitArray))
    if length(szin) != size(flag,1)
        error("Boolean flag must have as many rows as dimensions of A")
    end
    sz = Array(Int, size(flag))
    # First round of restriction comes from original size
    for i = 1:length(szin)
        sz[i] = flag[i] ? restrict_size(szin[i]) : szin[i]
    end
    # Remaining rounds of restriction come from the previous size
    for j = 2:size(flag, 2)
        for i = 1:size(flag, 1)
            sz[i,j] = flag[i,j] ? restrict_size(sz[i,j-1]) : sz[i,j-1]
        end
    end
    return sz
end
function prolong_size(szin::Union(Dims, Vector{Int}), flag::Union(Array{Bool},BitArray))
    sz = restrict_size(szin, flag)
    return [sz[:,end-1:-1:1] [szin...]]
end



## Utilities

to_tuple(v::Vector) = tuple(v...)
to_tuple(t::Union(Tuple, NTuple)) = t

# Apply a function to each dimension of an array, conditional upon a boolean
function mapdim(func::Function, A::Array, flag)
    if ndims(A) != size(flag,1)
        error("Boolean flag must have as many rows as dimensions of A")
    end
    Am = A
    for j = 1:size(flag, 2)
        for i = 1:size(flag, 1)
            if flag[i,j]
                Am = func(Am, i)
            end
        end
    end
    return Am
end

function filledges!(A::Array, val)
    n_dims = ndims(A)
    ind = Array(Any, n_dims)
    for idim = 1:n_dims
        ind[idim] = 1:size(A, idim)
    end
    for idim = 1:n_dims
        ind[idim] = [1, size(A, idim)]
        A[to_tuple(ind)...] = val
        ind[idim] = 1:size(A, idim)
    end
end

end # module
