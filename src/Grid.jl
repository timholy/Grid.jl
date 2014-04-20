module Grid

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
    Counter,
    AbstractInterpGrid,
    InterpGrid,
    InterpGridCoefs,
    InterpIrregular,
    InterpType,
    InterpNearest,
    InterpLinear,
    InterpQuadratic,
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
