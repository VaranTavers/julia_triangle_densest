module GeneticAlgorithm

using Random

struct GeneticSettings
    populationSize::Integer
    mutationRate::Float64
    crossoverRate::Float64
    elitRate::Float64
    crossoverAlg::Any
    mutationAlg::Any
    fitnessF::Any
end

struct RunSettings
    minDists::Any
    k::Integer
    numberOfIterations::Integer
    logging::Bool
    RunSettings(minDists, k, numberOfIterations) = new(minDists, k, numberOfIterations, "")
    RunSettings(minDists, k, numberOfIterations, logging) =
        new(minDists, k, numberOfIterations, logging)
end

sample(weights) = findfirst(cumsum(weights) .> rand())


function crossoverNaive(v1, v2)
    v3 = copy(v1)
    append!(v3, v2)
    v3Set = Set(v3)

    v3 = collect(v3Set)

    v3[randperm(length(v3))][1:length(v1)]
end

function crossoverRoulette(chromosomes, fitness, _minDists)
    rouletteWheel = fitness ./ sum(fitness)

    crossoverNaive(chromosomes[sample(rouletteWheel)], chromosomes[sample(rouletteWheel)])
end

function mutation(v, edgeMat)
    # TODO: Maybe less copy?
    vCopy = copy(v)
    index = rand(1:length(v))

    n, _ = size(edgeMat)
    toChange = rand(1:n)
    while toChange in vCopy
        toChange = rand(1:n)
    end

    vCopy[index] = toChange

    vCopy
end


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

function calculateFitnessHeavy(edgeMat, c)
    sum(snd.(calculateTriangles(edgeMat, c)))
end

function trianglesGenetic(
    runS::RunSettings,
    gaS::GeneticSettings,
    chromosomes::Vector{Vector{Int64}},
)
    # Initializing values and functions for later use
    n = length(chromosomes)
    numberOfPoints, _ = size(runS.minDists)
    calcFitness(x) = gaS.fitnessF(runS.minDists, x)
    runMutation(x) =
        rand() < gaS.mutationRate ? gaS.mutationAlg(x, runS.minDists) : x
    chromosomes = deepcopy(chromosomes)

    # Initializing global maximum as one of the given chromosome
    maxVal = calcFitness(chromosomes[1])
    maxVec = copy(chromosomes[1])

    # Initializing logging
    logs = []

    fitness = collect(map(calcFitness, chromosomes))
    for i = 1:runS.numberOfIterations
        # Creating p_c% new individuals with the crossover
        # operator, choosing parents based on fitness.
        newChromosomes = [
            gaS.crossoverAlg(chromosomes, fitness, runS.minDists) for
            _ = 1:Int(ceil(n * gaS.crossoverRate))
        ]
        newFitness = collect(map(calcFitness, chromosomes))

        # Add them to the chromosome pool
        append!(chromosomes, newChromosomes)
        append!(fitness, newFitness)

        # Mutating individuals
        chromosomes = collect(map(runMutation, chromosomes))

        # Recalculating fitness for new individuals
        fitness = collect(map(calcFitness, chromosomes))

        # Sorting fitness scores
        fitnessSorted = sortperm(fitness, rev=true)

        fitnessMaxVal = deepcopy(fitness[fitnessSorted[1]])
        fitnessMaxVec = deepcopy(chromosomes[fitnessSorted[1]])


        # Choosing the elit
        elitNumber = Int(ceil(gaS.populationSize * gaS.elitRate))
        elitChromosomes = deepcopy(chromosomes[fitnessSorted[1:elitNumber]])
        elitFitness = copy(fitness[fitnessSorted[1:elitNumber]])

        # Choosing the rest randomly from the others
        restNumber = gaS.populationSize - elitNumber
        restIds = [rand(fitnessSorted[elitNumber+1:end]) for _ = 1:restNumber]
        restChromosomes = map(x -> copy(chromosomes[x]), restIds)
        restFitness = map(x -> fitness[x], restIds)

        chromosomes = vcat(elitChromosomes, restChromosomes)
        fitness = vcat(elitFitness, restFitness)

        if fitnessMaxVal > maxVal
            maxVec = deepcopy(fitnessMaxVec)
            maxVal = deepcopy(fitnessMaxVal)
        end

        if runS.logging != ""
            logRow = [i, maxVal]
            append!(logRow, sort(maxVec))
            push!(logs, logRow)
        end
    end

    maxVec, logs
end

export GeneticSettings, RunSettings, trianglesGenetic, crossoverRoulette, mutation, calculateFitnessDense, calculateFitnessHeavy, calculateTriangles
end