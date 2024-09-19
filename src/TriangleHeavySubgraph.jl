module TriangleHeavySubgraph

include("acoKTriangles.jl")
include("geneticAlgorithm.jl")

using .GeneticAlgorithm
using .ACOKTriangles


# TODO: Maybe greedy?

import .GeneticAlgorithm: GeneticSettings, RunSettings, trianglesGenetic, mutation, crossoverRoulette, calculateFitnessDense, calculateFitnessHeavy, calculateTriangles, mutationNeighbor, crossoverRouletteBetter, crossoverRouletteSlow
import .ACOKTriangles: TrianglesACOK, TrianglesACOK_get_pheromone, TrianglesACOKEta, ACOKSettings, ACOSettings, solution_to_vec


export GeneticSettings, RunSettings, trianglesGenetic, mutation, crossoverRoulette, calculateFitnessDense, calculateFitnessHeavy, calculateTriangles, mutationNeighbor, crossoverRouletteBetterm, crossoverRouletteSlow
export TrianglesACOK, TrianglesACOK_get_pheromone, TrianglesACOKEta, ACOKSettings, ACOSettings, solution_to_vec


end
