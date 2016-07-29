import Base: copy!, eltype, getindex, isvalid, length, ndims, setindex!, show, size

#### Interpolation of evenly-spaced data ####

# This implements "generalized interpolation." See, for example,
# P. Thevenaz, T. Blu, and M. Unser (2000). "Interpolation Revisited."
# IEEE Transactions on Medical Imaging, 19: 739-758.

# This type manages temporary storage needed for efficient
# interpolation on a grid, so that once created it is possible to
# calculate many interpolated values without additional
# temporaries. It is designed to support both single- and multi-valued
# interpolation, and computation of the gradient with respect to the
# position of evaluation.
# An example of multi-valued interpolation is sub-pixel interpolation
# within an RGB image, and getting the 3 color values out.
type InterpGridCoefs{T, IT<:InterpType}
    coord1d::Vector{Vector{Int}}  # for 1-d positions
    coef1d::Vector{Vector{T}}     # for 1-d coefficients
    gcoef1d::Vector{Vector{T}}    # for 1-d coefficients of the gradient
    hcoef1d::Vector{Vector{T}}    # for 1-d coefficients of the hessian
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
abstract AbstractInterpGrid{T, N, BC<:BoundaryCondition, IT<:InterpType} <: AbstractArray{T,N}

type InterpGrid{T<:AbstractFloat, N, BC<:BoundaryCondition, IT<:InterpType} <: AbstractInterpGrid{T,N,BC,IT}
    coefs::Array{T,N}
    ic::InterpGridCoefs{T, IT}
    x::Vector{T}
    fillval::T  # used only for BCfill (if ever)
end
function InterpGrid{T<:AbstractFloat, N, BC<:BoundaryCondition, IT<:InterpType}(A::Array{T, N}, ::Type{BC}, ::Type{IT})
    if BC == BCfill
        error("Construct BCfill InterpGrids by supplying the fill value")
    end
    coefs = copy(A)
    interp_invert!(coefs, BC, IT, 1:N)
    ic = InterpGridCoefs(coefs, IT)
    x = zeros(T, N)
    InterpGrid{T, N, BC, IT}(coefs, ic, x, convert(T, NaN))
end
function InterpGrid{T<:AbstractFloat, N, BC<:Union{BCnil,BCnan,BCna}}(A::Array{T, N}, ::Type{BC}, ::Type{InterpCubic})
    # Cubic interpolation requires padding
    coefs=pad1(copy(A),0,1:N)
    interp_invert!(coefs, BC, InterpCubic, 1:N)
    ic = InterpGridCoefs(coefs, InterpCubic)
    x = zeros(T, N)
    InterpGrid{T, N, BC, InterpCubic}(coefs, ic, x, convert(T, NaN))
end
function InterpGrid{T<:AbstractFloat, N, IT<:InterpType}(A::Array{T, N}, f::Number, ::Type{IT})
    coefs = pad1(A, f, 1:N)
    interp_invert!(coefs, BCnearest, IT, 1:N)
    ic = InterpGridCoefs(coefs, IT)
    x = zeros(T, N)
    InterpGrid{T, N, BCfill, IT}(coefs, ic, x, convert(T, f))
end


setx{T}(G::InterpGrid{T,1}, x::Real) = @inbounds G.x[1] = x
function setx{T}(G::InterpGrid{T,2}, x::Real, y::Real)
    xG = G.x
    @inbounds xG[1] = x
    @inbounds xG[2] = y
end
function setx{T}(G::InterpGrid{T,3}, x::Real, y::Real, z::Real)
    xG = G.x
    @inbounds xG[1] = x
    @inbounds xG[2] = y
    @inbounds xG[3] = z
end
function setx{T,N}(G::InterpGrid{T,N}, x::Real...)
    if length(x) != N
        error("Incorrect number of dimensions supplied")
    end
    xG = G.x
    @inbounds for idim = 1:N
        xG[idim] = x[idim]
    end
end
# a version that corrects for the padding of BCfill types
setx{T}(G::InterpGrid{T,1,BCfill}, x::Real) = @inbounds G.x[1] = x+1
function setx{T}(G::InterpGrid{T,2,BCfill}, x::Real, y::Real)
    xG = G.x
    @inbounds xG[1] = x+1
    @inbounds xG[2] = y+1
end
function setx{T}(G::InterpGrid{T,3,BCfill}, x::Real, y::Real, z::Real)
    xG = G.x
    @inbounds xG[1] = x+1
    @inbounds xG[2] = y+1
    @inbounds xG[3] = z+1
end
function setx{T,N}(G::InterpGrid{T,N,BCfill}, x::Real...)
    if length(x) != N
        error("Incorrect number of dimensions supplied")
    end
    xG = G.x
    @inbounds for idim = 1:N
        xG[idim] = x[idim]+1
    end
end
# InterpCubic also needs to compensate for padding
setx{T,BC<:Union{BCna,BCnan,BCnil}}(G::InterpGrid{T,1,BC,InterpCubic}, x::Real) = @inbounds G.x[1] = x+1
function setx{T,BC<:Union{BCna,BCnan,BCnil}}(G::InterpGrid{T,2,BC,InterpCubic}, x::Real, y::Real)
    xG = G.x
    @inbounds xG[1] = x+1
    @inbounds xG[2] = y+1
end
function setx{T,BC<:Union{BCna,BCnan,BCnil}}(G::InterpGrid{T,3,BC,InterpCubic}, x::Real, y::Real, z::Real)
    xG = G.x
    @inbounds xG[1] = x+1
    @inbounds xG[2] = y+1
    @inbounds xG[3] = z+1
end
function setx{T,N,BC<:Union{BCna,BCnan,BCnil}}(G::InterpGrid{T,N,BC,InterpCubic}, x::Real...)
    if length(x) != N
        error("Incorrect number of dimensions supplied")
    end
    xG = G.x
    @inbounds for idim = 1:N
        xG[idim] = x[idim]+1
    end
end

## Evaluation at single points
function _getindex{T}(G::InterpGrid{T})
    set_position(G.ic, boundarycondition(G), false, false, G.x)
    interp(G.ic, G.coefs)
end
getindex{T}(G::InterpGrid{T,1}, x::Real) = (setx(G, x); _getindex(G))
function getindex{T}(G::InterpGrid{T,1}, x::Real, y::Real)
    if y != 1
        throw(BoundsError())
    end
    setx(G, x)
    _getindex(G)
end
getindex{T}(G::InterpGrid{T,2}, x::Real, y::Real) = (setx(G, x, y); _getindex(G))
getindex{T}(G::InterpGrid{T,3}, x::Real, y::Real, z::Real) = (setx(G, x, y, z); _getindex(G))
getindex{T,N}(G::InterpGrid{T,N}, x::Real...) = (setx(G, x...); _getindex(G))

function _valgrad{T,N}(g::AbstractVector{T}, G::InterpGrid{T,N})
    if length(g) != N
        error("Wrong number of components for the gradient")
    end
    ic = G.ic
    coefs = G.coefs
    set_position(ic, boundarycondition(G), true, false, G.x)
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
    set_position(ic, boundarycondition(G), true, false, G.x)
    val = interp(ic, coefs)
    set_gradient_coordinate(ic, 1)
    g = interp(ic, coefs)
    return val, g
end
function valgrad{T}(G::AbstractInterpGrid{T,1}, x::Real)
    setx(G, x)
    _valgrad(G)
end
function valgrad{T}(G::AbstractInterpGrid{T,2}, x::Real, y::Real)
    g = Array(T, 2)
    val = valgrad(g, G, x, y)
    return val, g
end
function valgrad{T}(G::AbstractInterpGrid{T,3}, x::Real, y::Real, z::Real)
    g = Array(T, 3)
    val = valgrad(g, G, x, y, z)
    return val, g
end
function valgrad{T}(G::AbstractInterpGrid{T}, x::Real...)
    g = Array(T, length(x))
    val = valgrad(g, G, x...)
    return val, g
end
valgrad{T}(g::AbstractVector{T}, G::InterpGrid{T}, x::Real) = (setx(G, x); _valgrad(g, G))
valgrad{T}(g::AbstractVector{T}, G::InterpGrid{T}, x::Real, y::Real) = (setx(G, x, y); _valgrad(g, G))
valgrad{T}(g::AbstractVector{T}, G::InterpGrid{T}, x::Real, y::Real, z::Real) = (setx(G, x, y, z); _valgrad(g, G))
valgrad{T}(g::AbstractVector{T}, G::InterpGrid{T}, x::Real...) = (setx(G, x...); _valgrad(g, G))

function _valgradhess{T,N}(g::AbstractVector{T}, h::AbstractMatrix{T}, G::InterpGrid{T,N})
    if length(g) != N
        error("Wrong number of components for the gradient")
    end
    if size(h,1) != N || size(h,2) != N
        error("Wrong number of components for the hessian")
    end
    ic = G.ic
    coefs = G.coefs
    set_position(ic, boundarycondition(G), true, true, G.x)
    val = interp(ic, coefs)
    for idim = 1:N
        set_gradient_coordinate(ic, idim)
        g[idim] = interp(ic, coefs)
        # diagonal elements of the hessian
        set_hessian_coordinate(ic, idim, idim)
        h[idim,idim] = interp(ic, coefs)
        # off-diagonal element. exploit the symmetry!
        for jdim = idim+1:N
            set_hessian_coordinate(ic, idim, jdim)
            h[idim,jdim] = interp(ic, coefs)
            h[jdim,idim] = h[idim,jdim]
        end
    end
    return val
end
function _valgradhess{T}(G::InterpGrid{T,1})
    ic = G.ic
    coefs = G.coefs
    set_position(ic, boundarycondition(G), true, true, G.x)
    val = interp(ic, coefs)
    set_gradient_coordinate(ic, 1)
    g = interp(ic, coefs)
    set_hessian_coordinate(ic, 1, 1)
    h = interp(ic, coefs)
    return val, g, h
end
function valgradhess{T}(G::AbstractInterpGrid{T,1}, x::Real)
    setx(G, x)
    _valgradhess(G)
end
function valgradhess{T}(G::AbstractInterpGrid{T,2}, x::Real, y::Real)
    g = Array(T, 2)
    h = Array(T, 2, 2)
    val = valgradhess(g, h, G, x, y)
    return val, g, h
end
function valgradhess{T}(G::AbstractInterpGrid{T,3}, x::Real, y::Real, z::Real)
    g = Array(T, 3)
    h = Array(T, 3, 3)
    val = valgradhess(g, h, G, x, y, z)
    return val, g, h
end
function valgradhess{T}(G::AbstractInterpGrid{T}, x::Real...)
    g = Array(T, length(x))
    h = Array(T, length(x), length(x))
    val = valgradhess(g, h, G, x...)
    return val, g, h
end
valgradhess{T}(g::Vector{T}, h::Matrix{T}, G::InterpGrid{T, 1}, x::Real) = (setx(G, x); _valgradhess(g, h, G))
valgradhess{T}(g::Vector{T}, h::Matrix{T}, G::InterpGrid{T, 2}, x::Real, y::Real) = (setx(G, x, y); _valgradhess(g, h, G))
valgradhess{T}(g::Vector{T}, h::Matrix{T}, G::InterpGrid{T, 3}, x::Real, y::Real, z::Real) = (setx(G, x, y, z); _valgradhess(g, h, G))
function valgradhess{T}(g::Vector{T}, h::Matrix{T}, G::InterpGrid{T}, x::Real...)
    setx(G, x...)
    _valgradhess(g, h, G)
end

## Vectorized evaluation at multiple points
function getindex{T,R<:Number}(G::AbstractInterpGrid{T,1}, x::AbstractVector{R})
    n = length(x)
    v = Array(T, n)
    for i = 1:n
        v[i] = getindex(G,x[i])
    end
    v
end
getindex{T,N,R<:Number}(G::AbstractInterpGrid{T,N}, x::AbstractVector{R}) = error("Linear indexing not supported")
function getindex{T,R<:Number}(G::AbstractInterpGrid{T,2}, x::AbstractVector{R}, y::AbstractVector{R})
    nx, ny = length(x), length(y)
    v = Array(T, nx, ny)
    for i = 1:nx
        for j = 1:ny
            v[i,j] = getindex(G, x[i], y[j])
        end
    end
    v
end
function getindex{T,N,R<:Number}(G::AbstractInterpGrid{T,N}, x::AbstractVector{R}, xrest::AbstractVector{R}...)
    if length(xrest) != N-1
        error("Dimensionality mismatch")
    end
    nx = length(x)
    nrest = [length(y) for y in xrest]
    v = Array(T, nx, nrest...)
    for c in Counter(nrest)
        for i = 1:nx
            setx(G, x[i], ntuple(j->xrest[j][c[j]], N-1)...)  # FIXME performance?? May not matter...
            v[i,c...] = _getindex(G)
        end
    end
    v
end


# This should work for the convert in the InterpIrregular constructor below,
# but for some reason it does not
#convert{T,S}(::Type{Array{T,1}}, r::Ranges{S}) = [convert(T, x) for x in r]

#### Non-uniform grid interpolation ####

# Currently supports only 1d, nearest-neighbor or linear
# Consequently, the internal representation may change in the future
# BCperiodic and BCreflect not supported
type InterpIrregular{T<:Number, S, N, BC<:BoundaryCondition, IT<:InterpType} <: AbstractInterpGrid{S,N,BC,IT}
    grid::Vector{Vector{T}}
    coefs::Array{S,N}
    x::Vector{T}
    fillval::S  # used only for BCfill (if ever)
end
InterpIrregular{T<:Number, BC<:BoundaryCondition, IT<:Union{InterpForward,InterpBackward,InterpNearest,InterpLinear}}(grid::Vector{T}, A::AbstractArray, ::Type{BC}, ::Type{IT}) =
    InterpIrregular(Vector{T}[grid], A, BC, IT) # special 1d syntax
InterpIrregular{T<:Number, BC<:BoundaryCondition, IT<:Union{InterpForward,InterpBackward,InterpNearest,InterpLinear}}(grid::(@compat Tuple{Vararg{Vector{T}}}), A::AbstractArray, ::Type{BC}, ::Type{IT}) =
    InterpIrregular(Vector{T}[grid...], A, BC, IT)
function InterpIrregular{T<:Number, S, N, BC<:BoundaryCondition, IT<:Union{InterpForward,InterpBackward,InterpNearest,InterpLinear}}(grid::Vector{Vector{T}}, A::AbstractArray{S, N}, ::Type{BC}, ::Type{IT})
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
    coefs = Array(S, size(A))
    copy!(coefs, A)
    x = zeros(T, N)
    InterpIrregular{T, S, N, BC, IT}(grid, coefs, x, convert(S, NaN))
end
function InterpIrregular{IT<:InterpType}(grid, A::Array, f::Number, ::Type{IT})
    iu = InterpIrregular(grid, A, BCfill, IT)
    iu.fillval = f
    iu
end

function _getindexii{T,S,BC<:Union{BCfill,BCna,BCnan}}(G::InterpIrregular{T,S,1,BC}, x::Number)
    g = G.grid[1]
    i = (x == g[1]) ? 2 : searchsortedfirst(g, x)
    (i == 1 || i == length(g)+1) ? G.fillval : _interpu(x, g, i, G.coefs, interptype(G))
end
function _getindexii{T,S}(G::InterpIrregular{T,S,1,BCnil}, x::Number)
    g = G.grid[1]
    i = (x == g[1]) ? 2 : searchsortedfirst(g, x)
    (i == 1 || i == length(g)+1) ? throw(BoundsError()) : _interpu(x, g, i, G.coefs, interptype(G))
end
function _getindexii{T,S}(G::InterpIrregular{T,S,1,BCnearest}, x::Number)
    g = G.grid[1]
    i = (x == g[1]) ? 2 : searchsortedfirst(g, x)
    i == 1 ? G.coefs[1] : i == length(g)+1 ? G.coefs[end] : _interpu(x, g, i, G.coefs, interptype(G))
end
# This next is necessary for precedence
getindex(G::InterpIrregular, x::Real) = _getindexii(G, x)
getindex(G::InterpIrregular, x::Number) = _getindexii(G, x)

_interpu(x, g, i, coefs, ::Type{InterpForward}) = coefs[i]
_interpu(x, g, i, coefs, ::Type{InterpBackward}) = coefs[i-1]
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
function set_position{T,BC<:BoundaryCondition,IT<:InterpType}(ic::InterpGridCoefs{T,IT}, ::Type{BC}, calc_grad::Bool, calc_hess::Bool, x::Vector{T})
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
    if !valid && BC == BCnil
        error("Invalid location")
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
            if calc_hess
                interp_hcoefs_1d(ic.hcoef1d[idim], BC, IT, dx)
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
set_position{T,BC<:BoundaryCondition,IT<:InterpType}(ic::InterpGridCoefs{T,IT}, ::Type{BC}, calc_grad::Bool, x::Vector{T}) = set_position(ic, BC, calc_grad, false, x)

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
    interp_coef(ic.coef, ic.c1d)
end
function set_hessian_coordinate(ic::InterpGridCoefs, i1::Int, i2::Int)
    if !ic.valid
        return
    end
    N = ndims(ic)
    if !(1 <= i1 <= N && 1 <= i2 <= N)
        error("Wrong dimension index")
    end
    for idim = 1:N
        if i1 == i2
            if idim == i1
                ic.c1d[idim] = ic.hcoef1d[idim]
            else
                ic.c1d[idim] = ic.coef1d[idim]
            end
        else
            if idim == i1 || idim == i2
                ic.c1d[idim] = ic.gcoef1d[idim]
            else
                ic.c1d[idim] = ic.coef1d[idim]
            end
        end
    end
    interp_coef(ic.coef, ic.c1d)
end

# "interp" evaluates the interpolation. Call "set_position" first. If
# you want to also evaluate the gradient, call
# "set_gradient_coordinate" followed by "interp" for each component.
function interp{T}(ic::InterpGridCoefs{T}, A::AbstractArray, index::Int)
    if !ic.valid
        return convert(T, NaN)
    end
    coef = ic.coef
    if ic.wrap
        offset = ic.index
        index -= 1
    else
        offset = ic.offset
        index += ic.offset_base
    end
    val = zero(T)
    for i = 1:length(coef)
        @inbounds c = coef[i]
        @inbounds val += c == zero(T) ? zero(T) : c*convert(T,A[offset[i]+index])
    end
    return convert(T, val)
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
function InterpGridCoefs{T,IT<:InterpType}(::Type{T}, ::Type{IT}, dims::Union{Dims,Vector{Int}}, strides::Union{Dims,Vector{Int}})
    N = length(strides)
    if length(dims) != N
        error("Length of dims and strides must match")
    end
    coord1d = Array(Vector{Int}, N)
    coef1d = Array(Vector{T}, N)
    gcoef1d = Array(Vector{T}, N)
    hcoef1d = Array(Vector{T}, N)
    c1d = Array(Vector{T}, N)
    l = npoints(IT)
    for idim = 1:N
        coord1d[idim] = Array(Int, l)
        coef1d[idim] = Array(T, l)
        gcoef1d[idim] = Array(T, l)
        hcoef1d[idim] = Array(T, l)
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
    InterpGridCoefs{T,IT}(coord1d,coef1d,gcoef1d,hcoef1d,c1d,[dims...],[strides...],offset,0,index,false,false,coef)
end
InterpGridCoefs{IT<:InterpType}(A::Array, ::Type{IT}) = InterpGridCoefs(eltype(A), IT, [size(A)...], [strides(A)...])



# Interpolation support routines

npoints(::Type{InterpForward}) = 1
npoints(::Type{InterpBackward}) = 1
npoints(::Type{InterpNearest}) = 1
npoints(::Type{InterpLinear}) = 2
npoints(::Type{InterpQuadratic}) = 3
npoints(::Type{InterpCubic}) = 4

# Test whether a given coordinate will yield an interpolated result
isvalid{BC<:BoundaryCondition,IT<:InterpType}(::Type{BC},::Type{IT}, x, len::Int) = true
isvalid{BC<:Union{BCnil,BCnan,BCna}}(::Type{BC}, ::Type{InterpForward}, x, len::Int) = x >= 0.0 && x <= len
isvalid{BC<:Union{BCnil,BCnan,BCna}}(::Type{BC}, ::Type{InterpBackward}, x, len::Int) = x >= 0.0 && x <= len
isvalid{BC<:Union{BCnil,BCnan,BCna}}(::Type{BC}, ::Type{InterpNearest}, x, len::Int) = x >= 0.5 && x-0.5 <= len
isvalid{BC<:Union{BCnil,BCnan,BCna}}(::Type{BC}, ::Type{InterpLinear}, x, len::Int) = x >= 1 && x <= len
isvalid{BC<:Union{BCnil,BCnan,BCna}}(::Type{BC}, ::Type{InterpQuadratic}, x, len::Int) = x >= 1 && x <= len
isvalid{BC<:Union{BCnil,BCnan,BCna}}(::Type{BC}, ::Type{InterpCubic}, x, len::Int) = 1 <= x <= len -1

# version for offsets
function interp_coords_1d(coord1d::Vector{Int}, ::Type{InterpForward})
    coord1d[1] = 0
end
# version for indices
function interp_coords_1d{BC<:BoundaryCondition}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpForward}, x, len::Int)
    ix = wrap(BC, round(Int, real(x)), len)
    coord1d[1] = ix
    return ix, zero(typeof(x)), false
end

# version for offsets
function interp_coords_1d(coord1d::Vector{Int}, ::Type{InterpBackward})
    coord1d[1] = 0
end
# version for indices
function interp_coords_1d{BC<:BoundaryCondition}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpBackward}, x, len::Int)
    ix = wrap(BC, round(Int, real(x)), len)
    coord1d[1] = ix
    return ix, zero(typeof(x)), false
end

# version for offsets
function interp_coords_1d(coord1d::Vector{Int}, ::Type{InterpNearest})
    coord1d[1] = 0
end
# version for indices
function interp_coords_1d{BC<:BoundaryCondition}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpNearest}, x, len::Int)
    ix = wrap(BC, round(Int, real(x)), len)
    coord1d[1] = ix
    return ix, zero(typeof(x)), false
end

# version for offsets
function interp_coords_1d(coord1d::Vector{Int}, ::Type{InterpLinear})
    coord1d[1] = 0
    coord1d[2] = 1
end
# version for indices
function interp_coords_1d{T,BC<:BoundaryCondition}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpLinear}, x::T, len::Int)
    ifx = floor(Int, x)
    dx = x-ifx
    ifx -= x < convert(T, ifx)
    ix = wrap(BC, ifx, len)
    @inbounds coord1d[1] = ix
    iswrap = (ix == len && dx > 0) || (ix == 1 && ifx < 1)
    @inbounds coord1d[2] = iswrap ? ix : wrap(BC, ix+1, len)
    return ix, dx, iswrap
end
function interp_coords_1d{T}(coord1d::Vector{Int}, ::Type{BCreflect}, ::Type{InterpLinear}, x::T, len::Int)
    ix = mod(floor(Int, x)-1, 2*len)
    dx = x-floor(Int, x)
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
    @inbounds coord1d[1] = ix
    @inbounds coord1d[2] = wrap(BCreflect, ixp, len)
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
    ix = round(Int, real(x))
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
function interp_coords_1d{BC<:Union{BCnil,BCnan,BCna}}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpQuadratic}, x, len::Int)
    if x > 1.5 && x+0.5 < len
        ix = round(Int, real(x))
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
function interp_coords_1d{T,BC<:Union{BCnearest,BCfill}}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpQuadratic}, x::T, len::Int)
    if x > 1.5 && x+0.5 < len
        ix = round(Int, real(x))
        coord1d[1] = ix-1
        coord1d[2] = ix
        coord1d[3] = ix+1
        iswrap = false
        dx = x-convert(T,ix)
    elseif x <= 1.5
        ix = 1
        coord1d[1] = 2
        coord1d[2] = 1
        coord1d[3] = 2
        iswrap = true
        if x < 1
            dx = zero(T)
        else
            dx = x-convert(T,ix)
        end
    else
        ix = len
        coord1d[1] = len-1
        coord1d[2] = len
        coord1d[3] = len-1
        iswrap = true
        if x > len
            dx = zero(T)
        else
            dx = x-convert(T,ix)
        end
    end
    return ix, dx, iswrap
end
function interp_coords_1d(coord1d::Vector{Int}, ::Type{BCreflect}, ::Type{InterpQuadratic}, x, len::Int)
    ix = round(Int, real(x))
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

# version for offsets
function interp_coords_1d(coord1d::Vector{Int}, ::Type{InterpCubic})
    coord1d[1] = -1
    coord1d[2] = 0
    coord1d[3] = 1
    coord1d[4] = 2
end
#versions for indices
function interp_coords_1d{BC<:BoundaryCondition}(coord1d::Vector{Int}, ::Type{BC}, ::Type{InterpCubic}, x, len::Int)
    ix = trunc(Int, real(x))
    dx = x - ix
    coord1d[2] = ix
    # the outermost non-wrapping indices are 2 and len-2
    iswrap = isvalid(BC, ix, 2, len-1)
    if iswrap
        coord1d[1] = wrap(BC, ix-1, len)
        coord1d[3] = wrap(BC, ix+1, len)
        coord1d[4] = wrap(BC, ix+2, len)
    else
        coord1d[1] = ix-1
        coord1d[3] = ix+1
        coord1d[4] = ix+2
    end
    return ix, dx, iswrap
end

function interp_coefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpForward}, dx::T)
    coef1d[1] = one(T)
end
function interp_gcoefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpForward}, dx::T)
    coef1d[1] = zero(T)
end

function interp_coefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpBackward}, dx::T)
    coef1d[1] = one(T)
end
function interp_gcoefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpBackward}, dx::T)
    coef1d[1] = zero(T)
end

function interp_coefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpNearest}, dx::T)
    coef1d[1] = one(T)
end
function interp_gcoefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpNearest}, dx::T)
    coef1d[1] = zero(T)
end

function interp_coefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpLinear}, dx::T)
    coef1d[1] = 1.0-dx
    coef1d[2] = dx
end
function interp_gcoefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpLinear}, dx::T)
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
function interp_hcoefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpQuadratic}, dx::T)
    coef1d[1] = one(T)
    coef1d[2] = -2*one(T)
    coef1d[3] = one(T)
end
function interp_coefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpCubic}, dx::T)
    coef1d[1] = (1-dx)^3/6
    coef1d[2] = 2/3-dx^2*(2-dx)/2
    coef1d[3] = 2/3-(dx-1)^2*(1+dx)/2
    coef1d[4] = dx^3/6
end
function interp_gcoefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpCubic}, dx::T)
    coef1d[1] = -(1-dx)^2/2
    coef1d[2] = 3dx^2/2-2dx
    coef1d[3] = 1/2+dx*(1-(3/2)*dx)
    coef1d[4] = dx^2/2
end
function interp_hcoefs_1d{T,BC<:BoundaryCondition}(coef1d::Vector{T}, ::Type{BC}, ::Type{InterpCubic}, dx::T)
    coef1d[1] = 1-dx
    coef1d[2] = 3dx-2
    coef1d[3] = 1-3dx
    coef1d[4] = dx
end

eltype{T, N, BC, IT}(G::InterpGrid{T, N, BC, IT}) = T
ndims{T, N, BC, IT}(G::InterpGrid{T, N, BC, IT}) = N
boundarycondition{T, N, BC, IT}(G::InterpGrid{T, N, BC, IT}) = BC
interptype{T, N, BC, IT}(G::InterpGrid{T, N, BC, IT}) = IT
size{T,N}(G::InterpGrid{T,N,BCfill}) = ntuple(i->size(G.coefs,i)-2, N)
size{T,N}(G::InterpGrid{T,N,BCfill}, i::Integer) = size(G.coefs, i)-2
size{T,N}(G::InterpGrid{T,N}) = size(G.coefs)
size{T,N}(G::InterpGrid{T,N}, i::Integer) = size(G.coefs, i)

eltype{T, S, N, BC, IT}(G::InterpIrregular{T, S, N, BC, IT}) = S
ndims{T, S, N, BC, IT}(G::InterpIrregular{T, S, N, BC, IT}) = N
boundarycondition{T, S, N, BC, IT}(G::InterpIrregular{T, S, N, BC, IT}) = BC
interptype{T, S, N, BC, IT}(G::InterpIrregular{T, S, N, BC, IT}) = IT
size(G::InterpIrregular) = size(G.coefs)
size(G::InterpIrregular, i::Integer) = size(G.coefs, i)

eltype{T,IT<:InterpType}(ic::InterpGridCoefs{T,IT}) = T
interptype{IT<:InterpType,T}(ic::InterpGridCoefs{T,IT}) = IT
ndims(ic::InterpGridCoefs) = length(ic.coord1d)
show(io::IO, ic::InterpGridCoefs) = print(io, "InterpGridCoefs{", eltype(ic), ",", interptype(ic), "}")


# Generalized interpolation of higher order than InterpLinear requires
# inversion of the interpolation operator. See, for example, the
# Thevanez citation above.

# We need efficient 1d slices:
immutable Slice1D{T,N} <: DenseArray{T,1}
    A::Array{T,N}
    index::StepRange{Int,Int}
end
size(S::Slice1D) = (length(S.index),)
size(S::Slice1D, d) = d==1 ? length(S.index) : 1
length(S::Slice1D) = length(S.index)

# Note: unsafe, no bounds-checking implemented
getindex{T}(S::Slice1D{T}, i::Real) = S.A[first(S.index)+(i-1)*step(S.index)]
getindex{T}(S::Slice1D{T}, i::Real, j::Real) = S[i]  # assume j=1, because a bounds-check will make it too slow
setindex!{T}(S::Slice1D{T}, x, i::Real) = (setindex!(S.A, x, first(S.index)+(i-1)*step(S.index)); S)
setindex!{T}(S::Slice1D{T}, x, i::Real, j::Real) = setindex!(S, x, i)

function copy!(dst::Vector, src::Slice1D)
    n = length(src)
    length(dst) == n || throw(DimensionMismatch("Length $(length(dst)) of destination does not equal length $n of source"))
    p = src.A
    rng = src.index
    j = first(rng)
    s = step(rng)
    for i = 1:n
        @inbounds dst[i] = p[j]
        j += s
    end
    dst
end

const Q3inv = [7/8 1/4 -1/8; -1/8 5/4 -1/8; -1/8 1/4 7/8] # for handling "snippets" of size 3 along each dimension (InterpQuadratic)
# This works in place. If instead it allocated the output for you, then
# calling multiple times (e.g., to apply inversions for different
# interptypes along different dimensions) would result in unnecessary
# allocations.
interp_invert!{BC<:BoundaryCondition}(A::Array, ::Type{BC}, ::Type{InterpForward}, dimlist) = A
interp_invert!{BC<:BoundaryCondition}(A::Array, ::Type{BC}, ::Type{InterpBackward}, dimlist) = A
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
            rng = range(coords2lin(cc, stridesA), stridesA[idim], n)
            A_ldiv_B!(M, Slice1D(A, rng))
        end
        sizeA[idim] = size(A, idim)
    end
    A
end
function interp_invert!{BC<:BoundaryCondition}(A::Array, ::Type{BC}, ::Type{InterpCubic}, dimlist)
    sizeA = [size(A)...]
    stridesA = [strides(A)...]

    for idim = dimlist
        n = size(A,idim)
        # Set up tridiagonal system: (c_i-1 + 4 c_i + c_i+1) / 6 = f_i
        du = fill(convert(eltype(A), 1/6), n-1)
        d = fill(convert(eltype(A), 4/6), n)
        dl = copy(du)
        M = _interp_invert_matrix(BC, InterpCubic, dl, d, du)
        sizeA[idim] = 1
        for cc in Counter(sizeA)
            rng = range(coords2lin(cc, stridesA), stridesA[idim], n)
            A_ldiv_B!(M, Slice1D(A, rng))
        end
        sizeA[idim] = size(A,idim)
    end
    A
end
interp_invert!{BC<:Union{BoundaryCondition,Number}}(A::Array, ::Type{BC}, IT) = interp_invert!(A, BC, IT, 1:ndims(A))
interp_invert!{BC<:Union{BoundaryCondition,Number}}(A::Array, ::Type{BC}, IT, dimlist...) = interp_invert!(A, BC, IT, dimlist)

function _interp_invert_matrix{BC<:Union{BCnil,BCnan,BCna},T}(::Type{BC}, ::Type{InterpQuadratic}, dl::Vector{T}, d::Vector{T}, du::Vector{T})
    # For these, the quadratic centered on x=2 is continued down to
    # 1 rather than terminating at 1.5
    n = length(d)
    d[1] = d[n] = 9/8
    dl[n-1] = du[1] = -1/4
    MT = lufact!(Tridiagonal(dl, d, du))
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
    M = lufact!(Tridiagonal(dl, d, du))
end
function _interp_invert_matrix{T}(::Type{BCperiodic}, ::Type{InterpQuadratic}, dl::Vector{T}, d::Vector{T}, du::Vector{T})
    n = length(d)
    MT = lufact!(Tridiagonal(dl, d, du))
    # Woodbury correction to wrap around
    U = zeros(T, n, 2)
    V = zeros(T, 2, n)
    C = zeros(T, 2, 2)
    C[1,1] = C[2,2] = 1/8
    U[1,1] = U[n,2] = 1
    V[1,n] = V[2,1] = 1
    M = Woodbury(MT, U, C, V)
end
function _interp_invert_matrix{T,BC<:Union{BCnearest,BCfill}}(::Type{BC}, ::Type{InterpQuadratic}, dl::Vector{T}, d::Vector{T}, du::Vector{T})
    n = length(d)
    du[1] += 1/8
    dl[n-1] += 1/8
    M = lufact!(Tridiagonal(dl, d, du))
end

function _interp_invert_matrix{BC<:Union{BCnil,BCnan,BCna},T}(::Type{BC}, ::Type{InterpCubic}, dl::Vector{T}, d::Vector{T}, du::Vector{T})
    # The most common way to terminate Cubic B-spline interpolations at the edges is by setting the second derivative to 0
    # Doing this yields c_1 = f_1 and c_N = f_N at the low and high end respectively
    # The BC also adds the equations a_0 = 2a_1 - a_2 for a ghost point a_0 outside the lower end of the domain,
    # and similarly for the higher end. These equations are represented in the top and bottom rows.
    d[1:2] = d[end-1:end] = 1
    du[1] = dl[end] = convert(T,-2)
    du[2] = dl[end-1] = du[end] = dl[1] = zero(T)
    MT = lufact!(Tridiagonal(dl, d, du))
    # Since the edge equations adds off-diagonal elemetns, we need Woodbury correction
    n = length(d)
    U = zeros(T, n, 2)
    V = zeros(T, 2, n)
    C = zeros(T, 2, 2)
    C[1,1] = C[2,2] = one(T)
    U[1,1] = U[n,2] = one(T)
    V[1,3] = V[2,n-2] = one(T)
    Woodbury(MT, U, C, V)
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
function interp_coef_generic{T}(coefs::Array{T}, coef1d::Vector{Vector{T}})
    n_dims = length(coef1d)
    l = length(coef1d[1])
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
        szv[i] = szv[i].+2
    end
    N = length(szt)
    ind = Array{Any}(N)
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
