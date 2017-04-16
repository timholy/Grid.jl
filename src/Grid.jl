isdefined(Base, :__precompile__) && __precompile__()

module Grid

using Compat
if VERSION > v"0.4.0-dev+3066"
    using WoodburyMatrices
end

const mp = module_parent(Grid)
if isdefined(mp, :Images) && isdefined(mp.Images, :restrict)
    import ..Images.restrict
end

if !isdefined(:range)
    range = Range
end

include("counter.jl")
include("boundaryconditions.jl")
include("interpflags.jl")
include("interp.jl")
include("restrict_prolong.jl")
include("utilities.jl")
include("coord.jl")


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
    BCinbounds,
    Counter,
    AbstractInterpGrid,
    InterpGrid,
    InterpGridCoefs,
    InterpIrregular,
    InterpType,
    InterpForward,
    InterpBackward,
    InterpNearest,
    InterpLinear,
    InterpQuadratic,
    InterpCubic,
    CoordInterpGrid,
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
    set_hessian_coordinate,
    set_position,
    set_size,
    valgrad,
    valgradhess,
    wrap

end # module
