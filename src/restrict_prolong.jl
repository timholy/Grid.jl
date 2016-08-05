#### Restrict/prolong operators ####

# These are operators for solving PDEs via multigrid. These are also
# very fast routines for changing the size of an array by twofold,
# using linear interpolation for enlarging, and properly anti-aliasing
# for shrinking.

typealias LapackScalar Union{Float32, Float64, Complex64, Complex128}
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
            startA = coords2lin(indices, sA)
            startP = coords2lin(indices, sP)
            # handle the on-grid points
            copy!(P, range(startP, 2*skipP, n), A, range(startA, skipA, n))
            # handle the off-grid points (linear interpolation)
            rP = range(startP+skipP, 2*skipP, n-1)
            Base.LinAlg.BLAS.axpy!(0.5, A, range(startA, skipA, n-1), P, rP)
            Base.LinAlg.BLAS.axpy!(0.5, A, range(startA+skipA, skipA, n-1), P, rP)
        end
    else
        # Interpolate at 1/4 and 3/4 points
        for indices = Counter(sz)
            startA = coords2lin(indices, sA)
            startP = coords2lin(indices, sP)
            rA1 = range(startA, skipA, n-1)
            rA2 = range(startA+skipA, skipA, n-1)
            rP1 = range(startP, 2*skipP, n-1)
            rP2 = range(startP+skipP, 2*skipP, n-1)
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
    sA = [strides(A)...]
    skipA = sA[dim]
    P = Array(typeof(0.5*A[1]+0.5*A[1+skipA]), to_tuple(sz))
    sP = [strides(P)...]
    skipP = sP[dim]
    n = size(A, dim)
    sz[dim] = 1   # prolonged dimension is handled directly
    if isodd(len)
        for indices = Counter(sz)
            iA = coords2lin(indices, sA)
            iP = coords2lin(indices, sP)
            for i = 1:n-1
                P[iP] = A[iA]
                P[iP+skipP] = lincomb(0.5, A[iA], 0.5, A[iA+skipA])
                iA += skipA
                iP += 2*skipP
            end
            P[iP] = A[iA]
        end
    else
        for indices = Counter(sz)
            iA = coords2lin(indices, sA)
            iP = coords2lin(indices, sP)
            for i = 1:n-1
                P[iP] = lincomb(0.75, A[iA], 0.25, A[iA+skipA])
                P[iP+skipP] = lincomb(0.25, A[iA], 0.75, A[iA+skipA])
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
            startA = coords2lin(indices, sA)
            startR = coords2lin(indices, sR)
            Base.LinAlg.BLAS.axpy!(scale, A, range(startA, 2*skipA, n), R, range(startR, skipR, n))
            rA = range(startA+skipA, 2*skipA, n-1)
            rR = range(startR, skipR, n-1)
            Base.LinAlg.BLAS.axpy!(scale/2, A, rA, R, rR)
            rR = range(startR+skipR, skipR, n-1)
            Base.LinAlg.BLAS.axpy!(scale/2, A, rA, R, rR)
        end
    else
        for indices = Counter(sz)
            startA = coords2lin(indices, sA)
            startR = coords2lin(indices, sR)
            rA = range(startA, 2*skipA, n-1)
            rR = range(startR, skipR, n-1)
            Base.LinAlg.BLAS.axpy!(0.75*scale, A, rA, R, rR)
            rA = range(startA+skipA, 2*skipA, n-1)
            Base.LinAlg.BLAS.axpy!(0.25*scale, A, rA, R, rR)
            rR = range(startR+skipR, skipR, n-1)
            Base.LinAlg.BLAS.axpy!(0.75*scale, A, rA, R, rR)
            rA = range(startA, 2*skipA, n-1)
            Base.LinAlg.BLAS.axpy!(0.25*scale, A, rA, R, rR)
        end
    end
    return R
end
# restrict for non-BLAS types
function restrict{T}(A::Array{T}, dim::Integer, scale::Real)
    sz = [size(A)...]
    sz[dim] = restrict_size(sz[dim])
    sA = [strides(A)...]
    skipA = sA[dim]
    R = Array(typeof(0.5*A[1]+0.5*A[1+skipA]), to_tuple(sz))
    sR = [strides(R)...]
    skipR = sR[dim]
    n = sz[dim]
    sz[dim] = 1
    if isodd(size(A, dim))
        for indices = Counter(sz)
            iA = coords2lin(indices, sA)
            iR = coords2lin(indices, sR)
            R[iR] = lincomb(scale, A[iA], scale/2, A[iA+skipA])
            for i = 2:n-1
                iA += 2*skipA
                iR += skipR
                R[iR] = lincomb(scale/2, A[iA-skipA], scale, A[iA], scale/2, A[iA+skipA])
            end
            R[iR+skipR] = lincomb(scale/2, A[iA+skipA], scale, A[iA+2*skipA])
        end
    else
        for indices = Counter(sz)
            iA = coords2lin(indices, sA)
            iR = coords2lin(indices, sR)
            R[iR] = lincomb(0.75*scale, A[iA], 0.25*scale, A[iA+skipA])
            iR += skipR
            for i = 2:n-1
                R[iR] = lincomb(0.25*scale, A[iA], 0.75*scale, A[iA+skipA], 0.75*scale, A[iA+2*skipA], 0.25*scale, A[iA+3*skipA])
                iR += skipR
                iA += 2*skipA
            end
            R[iR] = lincomb(0.25*scale, A[iA], 0.75*scale, A[iA+skipA])
        end
    end
    return R
end
restrict{T}(A::Array{T}, dim::Integer) = restrict(A, dim, one(T))
function restrict(A::Array, flag::Union{Array{Bool},BitArray}, scale::Real)
    func = (B, dim) -> restrict(B, dim, scale)
    return mapdim(func, A, flag)
end
restrict{T}(A::Array{T}, flag::Union{Array{Bool},BitArray}) = restrict(A, flag, one(eltype(T)))

# "Balanced" restriction and prolongation

# If you take an array of ones, restrict it, and then prolong it
# (i.e., compute P*R*A, where P is the prolongation operator and R is
# the restriction operator), it smooths the array. However, by
# comparison to all other points in the array the edges will tend
# towards zero. The "balanced" versions of restrict/prolong ensure
# that each point of the array is used equally, i.e., that sum(P*R, 1)
# is constant, and hence the edges will be preserved.
function restrictb{T}(A::Array{T}, dim::Integer, scale::Real)
    local a::T
    local b::T
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
            index = coords2lin(indices, s)
            indexr = coords2lin(indices, sr)
            Ar[indexr] = a*A[index] + b*A[index+Astep]
            Ar[indexr+offset_endr] = a*A[index+offset_end] + b*A[index+offset_end-Astep]
        end
    else
        b = scale/(2*sqrt(2))
        a = 3*b
        for indices = Counter(sz)
            index = coords2lin(indices, s)
            indexr = coords2lin(indices, sr)
            Ar[indexr] = a*A[index] + b*A[index+Astep]
            Ar[indexr+offset_endr] = a*A[index+offset_end] + b*A[index+offset_end-Astep]
        end
    end
    return Ar
end
restrictb{T}(A::Array{T}, dim::Integer) = restrictb(A, dim, one(T))
function restrictb(A::Array, flag::Union{Array{Bool},BitArray}, scale::Real)
    func = (B, dim) -> restrictb(B, dim, scale)
    return mapdim(func, A, flag)
end
restrictb{T}(A::Array{T}, flag::Union{Array{Bool},BitArray}) = restrictb(A, flag, one(eltype(T)))
function prolongb{T}(A::Array{T}, dim::Integer, len::Integer)
    local a::T
    local b::T
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
            index = coords2lin(indices, s)
            indexp = coords2lin(indices, sp)
            Ap[indexp] = a*A[index]
            Ap[indexp+Apstep] = b*A[index] + A[index+Astep]/2
            Ap[indexp+offset_endp] = a*A[index+offset_end]
            Ap[indexp+offset_endp-Apstep] = b*A[index+offset_end] + A[index+offset_end-Astep]/2
        end
    else
        b = 1/(2*sqrt(2))
        a = 3*b
        for indices = Counter(sz)
            index = coords2lin(indices, s)
            indexp = coords2lin(indices, sp)
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
            index = coords2lin(indices, s)
            indexr = coords2lin(indices, sr)
            Ar[indexr] = (2*scale)*A[index]
            Ar[indexr+offset_endr] = (2*scale)*A[index+offset_end]
        end
    else
        for indices = Counter(sz)
            index = coords2lin(indices, s)
            indexr = coords2lin(indices, sr)
            Ar[indexr] = (3*scale)*A[index] - scale*A[index+s[dim]]
            Ar[indexr+offset_endr] = (3*scale)*A[index+offset_end] - scale*A[index+offset_end-s[dim]]
        end
    end
    return Ar
end
restrict_extrap{T}(A::Array{T}, dim::Integer) = restrict_extrap(A, dim, one(T))
function restrict_extrap(A::Array, flag::Union{Array{Bool},BitArray}, scale::Real)
    func = (B, dim) -> restrict_extrap(B, dim, scale)
    return mapdim(func, A, flag)
end
restrict_extrap{T}(A::Array{T}, flag::Union{Array{Bool},BitArray}) = restrict_extrap(A, flag, one(T))

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
function restrict_size(szin::Union{Dims, Vector{Int}}, flag::Union{Array{Bool},BitArray})
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
function prolong_size(szin::Union{Dims, Vector{Int}}, flag::Union{Array{Bool},BitArray})
    sz = restrict_size(szin, flag)
    return [sz[:,end-1:-1:1] [szin...]]
end
