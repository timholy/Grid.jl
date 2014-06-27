import Grid
using Base.Test

ctarget = ([1,1],[2,1],[1,2],[2,2],[1,3],[2,3])
k = 0
for c in Grid.Counter((2,3))
    k += 1
    @test c == ctarget[k]
end
@test k == 6
k = 0
for c in Grid.Counter((3,0))
    k += 1
    @test c == ctarget[k]
end
@test k == 0
k = 0
for c in Grid.Counter(())
    k += 1
    @test c == ctarget[k]
end
@test k == 0
