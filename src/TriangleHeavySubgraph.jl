module TriangleHeavySubgraph

include("acoKTriangles.jl")
include("geneticAlgorithm.jl")

using .GeneticAlgorithm
using .ACOKTriangles


import .GeneticAlgorithm: GeneticSettings, RunSettings, trianglesGenetic, mutation, crossoverRoulette, calculateFitnessDense, calculateFitnessHeavy, calculateTriangles
import .ACOKTriangles: TrianglesACOK, TrianglesACOK_get_pheromone, TrianglesACOKEta, ACOKSettings, ACOSettings, solution_to_vec


export GeneticSettings, RunSettings, trianglesGenetic, mutation, crossoverRoulette, calculateFitnessDense, calculateFitnessHeavy, calculateTriangles
export TrianglesACOK, TrianglesACOK_get_pheromone, TrianglesACOKEta, ACOKSettings, ACOSettings, solution_to_vec


end
