module InterpNew

using Grid
import Grid: BoundaryCondition, InterpType
import Base: getindex, size

if VERSION.minor < 3
    using Cartesian
else
    using Base.Cartesian
end

type InterpGridNew{T, N, BC<:BoundaryCondition, IT<:InterpType} <: AbstractArray{T,N}
    coefs::Array{T,N}
    fillval::T
end
size(G::InterpGridNew, d::Integer) = size(G.coefs, d)
size(G::InterpGridNew) = size(G.coefs)

# Create bodies like these:
#    1 <= x_1 < size(G,1) || throw(BoundsError()))
#    1 <= x_2 < size(G,2) || throw(BoundsError()))
#    ix_1 = ifloor(x_1); fx_1 = x_1 - convert(typeof(x_1), ix_1)
#    ix_2 = ifloor(x_2); fx_2 = x_2 - convert(typeof(x_2), ix_2)
#    @inbounds ret =
#       (1-fx_1)*((1-fx_2)*A[ix_1, ix_2] + fx_2*A[ix_1, ix_2+1]) +
#       fx_1*((1-fx_2)*A[ix_1+1, ix_2] + fx_2*A[ix_1+1, ix_2+1])
#    ret
function body_gen(::Type{BCnil}, ::Type{InterpLinear}, N::Integer)
    ex = interplinear_gen(N)
    quote
        @nexprs $N d->(1 <= x_d <= size(G,d) || throw(BoundsError()))
        @nexprs $N d->(ix_d = ifloor(x_d); fx_d = x_d - convert(typeof(x_d), ix_d))
        @nexprs $N d->(ixp_d = ix_d == size(G,d) ? ix_d : ix_d + 1)
        A = G.coefs
        @inbounds ret = $ex
        ret
    end
end

function body_gen(::Type{BCnan}, ::Type{InterpLinear}, N::Integer)
    ex = interplinear_gen(N)
    quote
        @nexprs $N d->(1 <= x_d <= size(G,d) || return(nan(eltype(G))))
        @nexprs $N d->(ix_d = ifloor(x_d); fx_d = x_d - convert(typeof(x_d), ix_d))
        @nexprs $N d->(ixp_d = fx_d == 0 ? ix_d : ix_d + 1)
        A = G.coefs
        @inbounds ret = $ex
        ret
    end
end

function body_gen(::Type{BCna}, ::Type{InterpLinear}, N::Integer)
    ex = interplinear_gen(N)
    quote
        @nexprs $N d->(0 < x_d < size(G,d) + 1 || return(nan(eltype(G))))
        @nexprs $N d->(ix_d = ifloor(x_d); fx_d = x_d - convert(typeof(x_d), ix_d))
        @nexprs $N d->( if ix_d < 1
                            ix_d, ixp_d = 1, 1
                        elseif ix_d >= size(G,d)
                            ix_d, ixp_d = size(G,d), size(G,d)
                        else
                            ixp_d = ix_d+1
                        end)
        A = G.coefs
        @inbounds ret = $ex
        ret
    end
end

# mod is slow, so this goes to some effort to avoid two calls to mod
function body_gen(::Type{BCreflect}, ::Type{InterpLinear}, N::Integer)
    ex = interplinear_gen(N)
    quote
        @nexprs $N d->(ix_d = ifloor(x_d); fx_d = x_d - convert(typeof(x_d), ix_d))
        @nexprs $N d->( len = size(G,d);
                        if !(1 <= ix_d <= len)
                            ix_d = mod(ix_d-1, 2len)
                            if ix_d < len
                                ix_d = ix_d+1
                                ixp_d = ix_d < len ? ix_d+1 : len
                            else
                                ix_d = 2len-ix_d
                                ixp_d = ix_d > 1 ? ix_d-1 : 1
                            end
                        else
                            ixp_d = ix_d == len ? len : ix_d + 1
                        end)
        A = G.coefs
        @inbounds ret = $ex
        ret
    end
end

function body_gen(::Type{BCperiodic}, ::Type{InterpLinear}, N::Integer)
    ex = interplinear_gen(N)
    quote
        @nexprs $N d->(ix_d = ifloor(x_d); fx_d = x_d - convert(typeof(x_d), ix_d))
        @nexprs $N d->(ix_d = mod(ix_d-1, size(G,d)) + 1)
        @nexprs $N d->(ixp_d = ix_d < size(G,d) ? ix_d+1 : 1)
        A = G.coefs
        @inbounds ret = $ex
        ret
    end
end

function body_gen(::Type{BCnearest}, ::Type{InterpLinear}, N::Integer)
    ex = interplinear_gen(N)
    quote
        @nexprs $N d->(x_d = clamp(x_d, 1, size(G,d)))
        @nexprs $N d->(ix_d = ifloor(x_d); fx_d = x_d - convert(typeof(x_d), ix_d))
        @nexprs $N d->(ixp_d = ix_d == size(G,d) ? ix_d : ix_d+1)
        A = G.coefs
        @inbounds ret = $ex
        ret
    end
end

function body_gen(::Type{BCfill}, ::Type{InterpLinear}, N::Integer)
    ex = interplinear_gen(N)
    quote
        @nexprs $N d->(1 <= x_d <= size(G,d) || return(G.fillval))
        @nexprs $N d->(ix_d = ifloor(x_d); fx_d = x_d - convert(typeof(x_d), ix_d))
        @nexprs $N d->(ixp_d = ix_d == size(G,d) ? ix_d : ix_d + 1)
        A = G.coefs
        @inbounds ret = $ex
        ret
    end
end


function body_gen(::Type{BCinbounds}, ::Type{InterpLinear}, N::Integer)
    ex = interplinear_gen(N)
    quote
        @nexprs $N d->(ix_d = ifloor(x_d); fx_d = x_d - convert(typeof(x_d), ix_d))
        ixp_d = ix_d + 1
        A = G.coefs
        @inbounds ret = $ex
        ret
    end
end

# Here's the function that does the indexing magic. It works recursively.
# Try this:
#   interplinear_gen(1)
#   interplinear_gen(2)
# and you'll see what it does.
# indexfunc = (dimension, offset) -> expression,
#   for example  indexfunc(3, 0) returns  :ix_3
#                indexfunc(3, 1) returns  :(ix_3 + 1)
function interplinear_gen(N::Integer, offsets...)
    if length(offsets) < N
        sym = symbol("fx_"*string(length(offsets)+1))
        return :((one($sym)-$sym)*$(interplinear_gen(N, offsets..., 0)) + $sym*$(interplinear_gen(N, offsets..., 1)))
    else
        indexes = [offsets[d] == 0 ? symbol("ix_"*string(d)) : symbol("ixp_"*string(d))  for d = 1:N]
        return :(A[$(indexes...)])
    end
end


# For type inference
promote_type_grid(T, x...) = promote_type(T, typeof(x)...)

# Because interpolation expressions are naturally generated by recursion, it's best to
# have a body-generation function. That makes creating the functions a little more awkward,
# but not too bad. If you want to see what this does, just copy-and-paste what's inside
# the `eval` call at the command line (after importing all relevant names)

# This creates getindex
for IT in (InterpLinear,)
    # for BC in subtypes(BoundaryCondition)
    for BC in subtypes(BoundaryCondition)
        eval(ngenerate(:N, :(promote_type_grid(T, x...)), :(getindex{T,N}(G::InterpGridNew{T,N,$BC,$IT}, x::NTuple{N,Real}...)),
                      N->body_gen(BC, IT, N)))
    end
end

end
