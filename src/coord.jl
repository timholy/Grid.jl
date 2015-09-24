import Base.FloatRange

type CoordInterpGrid{T<:AbstractFloat, N, BC<:BoundaryCondition, IT<:InterpType, R} <: AbstractInterpGrid{T,N}
    coord::R
    grid::InterpGrid{T,N,BC,IT}
    function CoordInterpGrid(coord::NTuple{N,Range}, grid::InterpGrid{T,N,BC,IT})
        map(length,coord) == size(grid) || throw(DimensionMismatch("Coordinate lengths do not match grid size."))
        new(coord,grid)
    end
end

function CoordInterpGrid{T,N,BC,IT}(coord::NTuple{N,Range}, grid::InterpGrid{T,N,BC,IT})
    CoordInterpGrid{T,N,BC,IT,typeof(coord)}(coord,grid)
end

function CoordInterpGrid{T<:AbstractFloat,R<:Range}(coord::R, grid::InterpGrid{T,1})
    CoordInterpGrid((coord,),grid)
end

function CoordInterpGrid{N,T<:AbstractFloat}(coord::NTuple{N,Range}, A::Array{T,N}, args...)
    CoordInterpGrid(coord,InterpGrid(A,args...))
end
function CoordInterpGrid{R<:Range,T<:AbstractFloat}(coord::R,A::Array{T,1},args...)
    CoordInterpGrid((coord,),InterpGrid(A,args...))
end


size(C::CoordInterpGrid) = size(C.grid)

coordlookup(r::FloatRange,x::Real) = (r.divisor*x-r.start)/r.step + 1.0
coordlookup(r::StepRange,x::Real) = (x-r.start)/r.step + 1.0
coordlookup(r::UnitRange,x::Real) = x-r.start + 1
coordlookup{N}(r::NTuple{N},x::NTuple{N}) = map(coordlookup,r,x)

function getindex{T}(C::CoordInterpGrid{T,1}, x::Real)
    getindex(C.grid,coordlookup(C.coord[1],x))
end
function getindex{T}(C::CoordInterpGrid{T,2}, x::Real, y::Real)
    getindex(C.grid,coordlookup(C.coord[1],x),coordlookup(C.coord[2],y))
end
function getindex{T}(C::CoordInterpGrid{T,3}, x::Real, y::Real, z::Real)
    getindex(C.grid,coordlookup(C.coord[1],x),coordlookup(C.coord[2],y),coordlookup(C.coord[3],z))
end
function getindex{T}(C::CoordInterpGrid{T,4}, x::Real, y::Real, z::Real, q::Real)
    getindex(C.grid,coordlookup(C.coord[1],x),coordlookup(C.coord[2],y),coordlookup(C.coord[3],z),coordlookup(C.coord[4],q))
end
function getindex{T,N}(C::CoordInterpGrid{T,N}, x::Real...)
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
