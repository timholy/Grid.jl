module Grid

const mp = module_parent(Grid)
if isdefined(mp, :Images) && isdefined(mp.Images, :restrict)
    import ..Images.restrict
end

include("counter.jl")
include("boundaryconditions.jl")
include("interpflags.jl")
include("interp.jl")
include("restrict_prolong.jl")
include("utilities.jl")

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

end # module
