# Estimate derivatives numerically
# These use centered-differencing for higher accuracy
function derivative_numer{T<:Number}(func::Function, x::T, h::T)
    xp = x + h
    vp = func(xp)
    xm = x - h
    vm = func(xm)
    return (vp-vm)/(xp-xm)
end
function derivative2_numer{T<:Number}(func::Function, x::T, h::T)
    return (func(x + 2h) - 2*func(x) + func(x - 2h)) / 4h^2
end
derivative_numer{T<:Number}(func::Function, x::T, index::Int, h::T) = derivative_numer(func, x, h)  # compatibility with partial derivatives
derivative2_numer{T<:Number}(func::Function, x::T, index::Int, h::T) = derivative2_numer(func, x, h)
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
function derivative2_numer{T<:Number}(func::Function, x::Array{T}, index::(@compat Tuple{Int,Int}), h::T)
    if index[1] == index[2] # 2nd order derivative in one direction
        v0 = func(x...)
        xsave = x[index[1]]
        xp = xsave + 2h
        xm = xsave - 2h
        x[index[1]] = xp
        vp = func(x...)
        x[index[1]] = xm
        vm = func(x...)
        x[index[1]] = xsave
        deriv = (vp - 2v0 + vm) / 4h^2
    else # cross-derivative, 1st order in each direction
        xpp = [x[1]+h,x[2]+h]
        xpm = [x[1]+h,x[2]-h]
        xmp = [x[1]-h,x[2]+h]
        xmm = [x[1]-h,x[2]-h]
        deriv = (func(xpp...) - func(xpm...) - func(xmp...) + func(xmm...)) / 4h^2
    end
    return deriv
end
function derivative_numer{T<:Number}(func::Function, c::(@compat Tuple{Vararg{T}}), index::Int, h::T)
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
function derivative2_numer{T<:Number}(func::Function, c::(@compat Tuple{Vararg{T}}), index::(@compat Tuple{Int,Int}), h::T)
    x = [c...]
    return derivative2_numer(func, x, index, h)
end
function derivative_numer{T<:Number}(func::Function, x::T, h::Vector{T})
    d = zeros(T, length(h))
    for i = 1:length(h)
        d[i] = derivative_numer(func, x, h[i])
    end
    return d
end
function derivative2_numer{T<:Number}(func::Function, x, h::Vector{T})
    d = zeros(T, length(h))
    for i = 1:length(h)
        d[i] = derivative2_numer(func, x, h[i])
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
function derivative2_numer{T<:Number}(func::Function, x, index::(@compat Tuple{Int,Int}), h::Vector{T})
    d = zeros(T, length(h))
    for i = 1:length(h)
        d[i] = derivative2_numer(func, x, index, h[i])
    end
    return d
end
derivative_numer{T<:Number}(func::Function, x::T) = derivative_numer(func, x, (eps(max(abs(x),one(T))))^convert(T, 1/3))
derivative_numer{T<:Number}(func::Function, x::Array{T}, index::Int) = derivative_numer(func, x, index, (eps(max(abs(x[index]),one(T))))^convert(T, 1/3))
derivative_numer{T<:Number}(func::Function, c::(@compat Tuple{Vararg{T}}), index::Int) = derivative_numer(func, c, index, (eps(max(abs(c[index]),one(T))))^convert(T, 1/3))
derivative2_numer{T<:Number}(func::Function, x::T) = derivative2_numer(func, x, (eps(max(abs(x),one(T))))^convert(T,1/3))
derivative2_numer{T<:Number}(func::Function, x::Array{T}, index::(@compat Tuple{Int,Int})) = derivative2_numer(func, x, index, (eps(max(abs(x[index[1]]),one(T))))^convert(T, 1/3))
derivative2_numer{T<:Number}(func::Function, c::(@compat Tuple{Vararg{T}}), index::(@compat Tuple{Int,Int})) = derivative2_numer(func, c, index, (eps(max(abs(c[index[1]]),one(T))))^convert(T, 1/3))
