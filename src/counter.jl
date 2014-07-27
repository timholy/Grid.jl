import Base: done, next, start

#### Iterator ####
# An n-dimensional grid iterator. This allows you to do things in
# arbitrary dimensions that in fixed dimensions you might do with a
# comprehension.
#
# For example, for two dimensions the equivalent of
#    p = [ i*j for i = 1:m, j = 1:n ]
# is
#    sz = (m, n)
#    p = Array(Int, sz)
#    index = 1
#    for c = Counter(sz)
#      p[index] = prod(c)
#      index += 1
#    end
# However, with sz appropriately defined, this version works for
# arbitrary dimensions.

immutable Counter
    max::Vector{Int}
end
Counter(sz::Tuple) = Counter(Int[sz...])

function start(c::Counter)
    N = length(c.max)
    state = ones(Int,N)
    if N > 0
        state[1] = 0 # because of start/done/next sequence, start out one behind
    end
    return state
end
function done(c::Counter, state)
    if isempty(state)
        return true
    end
    # we do the increment as part of "done" to make exit-testing more efficient
    state[1] += 1
    i = 1
    max = c.max
    while state[i] > max[i] && i < length(state)
        state[i] = 1
        i += 1
        state[i] += 1
    end
    state[end] > max[end]
end
next(c::Counter, state) = state, state
