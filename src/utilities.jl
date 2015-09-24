## Utilities

to_tuple(v::Vector) = tuple(v...)
to_tuple(t::Union{Tuple, NTuple}) = t

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

# Devectorized linear combinations
lincomb(coef1, A1, coef2, A2) = coef1*A1 + coef2*A2
lincomb(coef1, A1, coef2, A2, coef3, A3) = coef1*A1 + coef2*A2 + coef3*A3
lincomb(coef1, A1, coef2, A2, coef3, A3, coef4, A4) = coef1*A1 + coef2*A2 + coef3*A3 + coef4*A4
function lincomb(coef1::Number, A1::AbstractArray, coef2::Number, A2::AbstractArray)
    out = similar(A1)
    for i = 1:length(A1)
        out[i] = coef1*A1[i] + coef2*A2[i]
    end
    out
end

function lincomb(coef1::Number, A1::AbstractArray, coef2::Number, A2::AbstractArray, coef3::Number, A3::AbstractArray)
    out = similar(A1)
    for i = 1:length(A1)
        out[i] = coef1*A1[i] + coef2*A2[i] + coef3*A3[i]
    end
    out
end

function lincomb(coef1::Number, A1::AbstractArray, coef2::Number, A2::AbstractArray, coef3::Number, A3::AbstractArray, coef4::Number, A4::AbstractArray)
    out = similar(A1)
    for i = 1:length(A1)
        out[i] = coef1*A1[i] + coef2*A2[i] + coef3*A3[i] + coef4*A4[i]
    end
    out
end


# Converting coordinates to linear indices
function coords2lin(c, strides)
    ind = 1
    for i = 1:length(c)
        ind += (c[i]-1)*strides[i]
    end
    ind
end
