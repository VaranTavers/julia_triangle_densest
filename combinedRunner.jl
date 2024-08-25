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

using TriangleHeavySubgraph

name = "W_moreno_lesmis_lesmis"

fst((x, _)) = x


plotting = true
logging = true
numberOfRuns = 5
maxFitnessEvals = 10000
files = readdir("graphs/")[6:6]
ks = [25, 50, 100]

df = DataFrame(k=Int[], graph=String[], gaVal=Float64[], gaRes=String[], acoVal=Float64[], acoRes=String[])

function edgeInTriangles(edge, triangles)
    for (nodes, _) in triangles
        if src(edge) in nodes && dst(edge) in nodes
            return true
        end
    end

    false
end

function getAllKCombinations!(res, current, vals, i, k)
    for (j, v) in enumerate(vals)
        current[i] = v
        if i == k
            push!(res, copy(current))
        else
            getAllKCombinations!(res, current, vals[(j+1):end], i + 1, k)
        end
    end
end


# Default settings for genetic algorithm
gaS = GeneticSettings(100, 0.2, 0.8, 0.5, crossoverRoulette, mutation, calculateFitnessDense)

# Default settings for ACO algorithm
vars = ACOSettings(
    1.5, # α
    2, # β
    100, # number_of_ants
    0.7, # ρ
    0.005, # ϵ
    maxFitnessEvals / 2, # maxNumberOfEvals
    300, # starting_pheromone_ammount
    calculateFitnessDense,
    (x) -> collect(Set(x))
)


l_files = length(files)

df = DataFrame(
    graphs=files,
    meanGA=zeros(l_files),
    stdGA=zeros(l_files),
    minGA=zeros(l_files),
    maxGA=zeros(l_files),
    meanACO=zeros(l_files),
    stdACO=zeros(l_files),
    minACO=zeros(l_files),
    maxACO=zeros(l_files),
)

if !isdir("./logs")
    mkdir("./logs")
end
if !isdir("./results")
    mkdir("./results")
end

for k in ks
    conf_name = "$(k)"
    println("$(k) started on $(Dates.now())")

    date_of_start = Dates.today()
    if !isdir("logs/$(conf_name)_$(date_of_start)")
        mkdir("logs/$(conf_name)_$(date_of_start)")
    end
    if !isdir("results/$(conf_name)_$(date_of_start)")
        mkdir("results/$(conf_name)_$(date_of_start)")
    end
    open("results/$(conf_name)_$(date_of_start)/params.txt", "a") do io
        println(io, "k=", k)
        println(io, "maxFitnessEvals=", maxFitnessEvals)
    end

    for (i, f) in enumerate(files)
        println("$(i)/$(length(files)) $(f) started on $(Dates.now())")

        # Loading graph
        graph = loadgraph("graphs/$(f)", WELFormat(' '))
        n = nv(graph)
        edgeMat = graph.weights

        if n < k
            continue
        end

        # Running ACO Algorithm
        vars3 = ACOKSettings(
            vars,
            k,
            false,
            Integer(ceil(k * 1.5))
        )
        @time resultsACO = Folds.map(_ -> TrianglesACOK(edgeMat, vars3; logging=logging), 1:numberOfRuns)
        ###

        @show fst.(resultsACO)
        # Running Genetic Algorithm
        runS = RunSettings(edgeMat, k, maxFitnessEvals / 2, logging)
        @time resultsGA = Folds.map(x -> trianglesGenetic(
                runS,
                gaS,
                [i < gaS.populationSize * 0.25 ? x : randperm(n)[1:k] for i in 1:gaS.populationSize],
            ), fst.(resultsACO))


        valuesGA = Folds.map(((x, y),) -> calculateFitnessDense(edgeMat, x), resultsGA)
        valuesACO = Folds.map(((x, y),) -> calculateFitnessDense(edgeMat, x), resultsACO)

        if logging
            max_valGA = argmax(valuesGA)
            _, logs = resultsGA[max_valGA]
            CSV.write(
                "logs/$(conf_name)_$(date_of_start)/logs_$(f)_$(max_valGA)_GA.csv",
                DataFrame(logs, :auto),
            )
            max_valACO = argmax(valuesACO)
            _, logs = resultsACO[max_valACO]
            CSV.write(
                "logs/$(conf_name)_$(date_of_start)/logs_$(f)_$(max_valACO)_ACO.csv",
                DataFrame(logs, :auto),
            )
        end

        if plotting && nv(graph) < 500
            if !isdir("images/$(conf_name)_$(date_of_start)")
                mkdir("images/$(conf_name)_$(date_of_start)")
            end
            layout = circular_layout
            colors = distinguishable_colors(3)

            for (runNum, res) in enumerate(fst.(resultsGA))
                trianglesGA = calculateTriangles(edgeMat, res)

                heaviestCommunity = [i in res !== nothing ? 1 : 0 for i in 1:nv(graph)]
                edgeColors = [edgeInTriangles(e, trianglesGA) ? 3 : 1 for e in edges(graph)]
                plot = gplot(graph, nodesize=50, layout=layout,
                    nodelabel=1:nv(graph),
                    nodefillc=colors[heaviestCommunity.+2],
                    edgestrokec=colors[edgeColors],
                    nodestrokelw=3,
                    NODELABELSIZE=2,
                    plot_size=(16cm, 16cm)
                )

                saveplot(plot, "images/$(conf_name)_$(date_of_start)/$(f)_run$(runNum)_GA.svg")

            end

            for (runNum, res2) in enumerate(fst.(resultsACO))
                trianglesACO = calculateTriangles(edgeMat, res2)

                heaviestCommunity2 = [i in res2 !== nothing ? 1 : 0 for i in 1:nv(graph)]

                edgeColors = [edgeInTriangles(e, trianglesACO) ? 3 : 1 for e in edges(graph)]
                plot2 = gplot(graph, nodesize=50, layout=layout,
                    nodelabel=1:nv(graph),
                    nodefillc=colors[heaviestCommunity2.+2],
                    edgestrokec=colors[edgeColors],
                    nodestrokelw=3,
                    NODELABELSIZE=2,
                    plot_size=(16cm, 16cm)
                )

                saveplot(plot2, "images/$(conf_name)_$(date_of_start)/$(f)_run$(runNum)_ACO.svg")
            end
        end

        CSV.write(
            "results/$(conf_name)_$(date_of_start)/run_result_$(f)_GA.csv",
            DataFrame(fst.(resultsGA), :auto),
        )
        CSV.write(
            "results/$(conf_name)_$(date_of_start)/run_result_$(f)_ACO.csv",
            DataFrame(fst.(resultsACO), :auto),
        )
        df[i, "maxGA"] = maximum(valuesGA)
        df[i, "minGA"] = minimum(valuesGA)
        df[i, "stdGA"] = std(valuesGA)
        df[i, "meanGA"] = mean(valuesGA)
        df[i, "maxACO"] = maximum(valuesACO)
        df[i, "minACO"] = minimum(valuesACO)
        df[i, "stdACO"] = std(valuesACO)
        df[i, "meanACO"] = mean(valuesACO)
        if df[1, 6] != 0 || df[1, 2] != 0
            CSV.write("results/$(conf_name)_$(date_of_start)/results_compiled.csv", df)
        end
    end
end


println("Alg_tester_cli ended on $(Dates.now())")
