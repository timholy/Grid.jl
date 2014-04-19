import Base.FloatRange

type CoordInterpGrid{T<:FloatingPoint, N, BC<:BoundaryCondition, IT<:InterpType, R<:Range} <: AbstractInterpGrid{T,N}
    coord::NTuple{N,R}
    grid::InterpGrid{T,N,BC,IT}
end

function CoordInterpGrid{T<:FloatingPoint,R<:Range}(coord::R, grid::InterpGrid{T,1})
    CoordInterpGrid((coord,),grid)
end

function CoordInterpGrid{N,R<:Range,T<:FloatingPoint}(coord::NTuple{N,R}, A::Array{T,N}, args...)
    CoordInterpGrid(coord,InterpGrid(A,args...))
end
function CoordInterpGrid{R<:Range,T<:FloatingPoint}(coord::R,A::Array{T,1},args...)
    CoordInterpGrid((coord,),InterpGrid(A,args...))
end


size(C::CoordInterpGrid) = size(C.grid)

coordlookup(r::FloatRange,x::Real) = (r.divisor*x-r.start)/r.step + 1.0
coordlookup(r::StepRange,x::Real) = (x-r.start)/r.step + 1.0
coordlookup(r::UnitRange,x::Real) = x-r.start + 1
coordlookup{N,R<:Range,T<:Real}(r::NTuple{N,R},x::NTuple{N,T}) = map(coordlookup,r,x)

function getindex(C::CoordInterpGrid, x::Real...)
    getindex(C.grid,coordlookup(C.coord,x)...)
end

# gradients
# chain rule: dy/dx = (dy/du) / (dx/du)

coordiscale(r::FloatRange,x::Real) = x*r.divisor/r.step
coordiscale(r::StepRange,x::Real) = x/r.step
coordiscale(r::UnitRange,x::Real) = x

function valgrad{T,BC,IT,R}(C::CoordInterpGrid{T,1,BC,IT,R}, x::Real)
    val, g = valgrad(C.grid,coordlookup(C.coord[1],x))
    val, coordiscale(C.coord[1],g)
end

function valgrad{T,N,BC,IT,R}(g::Vector{T}, C::CoordInterpGrid{T,N,BC,IT,R}, x::Real...)
    val = valgrad(g,C.grid,coordlookup(C.coord,x)...)
    for i=1:length(g)
        g[i] = coordiscale(C.coord[i],g[i])
    end
    val
end
