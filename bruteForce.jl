using Pkg
using Test

Pkg.activate(".")

using Graphs
using SimpleWeightedGraphs

import Cairo
import Fontconfig

using Colors
using GraphPlot
using Compose
using Random
using WeightedEdgeListFormat
using DataFrames
using CSV
using Dates
using Folds
using Statistics


files = readdir("graphs/")[9:9]
ks = [25]

global maxVal = 0
global maxNodes = []

function calculateTriangles(edgeMat, c)
    res = Set()

    for node1 in c
        for node2 in c
            if edgeMat[node1, node2] != 0
                for node3 in c
                    if edgeMat[node2, node3] != 0 && edgeMat[node3, node1] != 0
                        push!(res, (sort!([node1, node2, node3]), edgeMat[node1, node2] + edgeMat[node2, node3] + edgeMat[node3, node1]))
                    end
                end
            end
        end
    end

    collect(res)
end

function calculateFitnessDense(edgeMat, c)
    length(calculateTriangles(edgeMat, c))
end


mutable struct Result
    maxVal::Int64
    maxNodes::Vector{Int64}
end

function calculateAllKCombinations!(res, edgeMat, current, vals, i, k)
    for (j, v) in enumerate(vals)
        current[i] = v
        if i == k
            cVal = calculateFitnessDense(edgeMat, current[1:k])

            if cVal > res.maxVal
                #@show maxVal, maxNodes, cVal, current
                res.maxVal = cVal
                res.maxNodes = deepcopy(current)
            end
        else
            calculateAllKCombinations!(res, edgeMat, current, vals[(j+1):end], i + 1, k)
        end
    end

    res
end

for k in ks
    for (i, f) in enumerate(files)
        println("$(i)/$(length(files)) $(f) started on $(Dates.now())")

        # Loading graph
        graph = loadgraph("graphs/$(f)", WELFormat(' '))
        n = nv(graph)
        edgeMat = graph.weights

        if n < k
            continue
        end


        ress = Folds.map(x -> calculateAllKCombinations!(Result(0, []), edgeMat, vcat([x], Int.(zeros(k - 1))), collect(x+1:n), 2, k), 1:(n-k+1))

        @show ress

        maxId = argmax(map(x -> x.maxVal, ress))

        @show ress[maxId]
    end
end