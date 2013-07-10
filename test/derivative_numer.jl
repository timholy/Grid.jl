# Estimate derivatives numerically
# These use centered-differencing for higher accuracy
function derivative_numer{T<:Number}(func::Function, x::T, h::T)
    xp = x + h
    vp = func(xp)
    xm = x - h
    vm = func(xm)
    return (vp-vm)/(xp-xm)
end
derivative_numer{T<:Number}(func::Function, x::T, index::Int, h::T) = derivative_numer(func, x, h)  # compatibility with partial derivatives
function derivative_numer{T<:Number}(func::Function, x::Array{T}, index::Int, h::T)
    xsave = x[index]
    xp = xsave + h
    xm = xsave - h
    x[index] = xp
    vp = func(x)
    x[index] = xm
    vm = func(x)
    x[index] = xsave
    return (vp-vm)/(xp-xm)
end
function derivative_numer{T<:Number}(func::Function, c::(T...,), index::Int, h::T)
    x = [c...]
    xsave = x[index]
    xp = xsave + h
    xm = xsave - h
    x[index] = xp
    vp = func(x...)
    x[index] = xm
    vm = func(x...)
    x[index] = xsave
    return (vp-vm)/(xp-xm)
end
function derivative_numer{T<:Number}(func::Function, x::T, h::Vector{T})
    d = zeros(T, length(h))
    for i = 1:length(h)
        d[i] = derivative_numer(func, x, h[i])
    end
    return d
end
function derivative_numer{T<:Number}(func::Function, x, index::Int, h::Vector{T})
    d = zeros(T, length(h))
    for i = 1:length(h)
        d[i] = derivative_numer(func, x, index, h[i])
    end
    return d
end
derivative_numer{T<:Number}(func::Function, x::T) = derivative_numer(func, x, (eps(max(abs(x),one(T))))^convert(T, 1/3))
derivative_numer{T<:Number}(func::Function, x::Array{T}, index::Int) = derivative_numer(func, x, index, (eps(max(abs(x[index]),one(T))))^convert(T, 1/3))
derivative_numer{T<:Number}(func::Function, c::(T...,), index::Int) = derivative_numer(func, c, index, (eps(max(abs(c[index]),one(T))))^convert(T, 1/3))
