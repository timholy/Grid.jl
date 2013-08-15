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
