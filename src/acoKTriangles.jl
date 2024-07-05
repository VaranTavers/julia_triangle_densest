module ACOKTriangles

begin
    using Graphs
    using SimpleWeightedGraphs
    using Folds
end

struct ACOSettings
    α::Real
    β::Real
    number_of_ants::Integer
    ρ::Real
    ϵ::Real
    maxNumberOfEvals::Integer
    starting_pheromone_ammount::Real
    eval_f::Function
    compute_solution::Function
    ACOSettings(α, β, n_a, ρ, ϵ, max_i, start_ph) = new(α, β, n_a, ρ, ϵ, max_i, start_ph, (_, _) -> 1.0, (_, _) -> 1.0)
    ACOSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s) = new(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s)
end

struct ACOKSettings
    acos # :: ACOSettings
    k::Real
    # There are situations when the ACO algorithm is unable to create the k subgraph
    # There is two options there:
    # 	false - skip the solution (faster, but might give worse answers, this is recommended if you have points with no neighbours)
    # 	true  - regenerate solution until a possible one is created (slower, but might give better answers)
    force_every_solution::Bool
    # If we force the length to be exactly k we may not get the correct answer since if we enter into a point from which only negative edges lead to anywhere the algorithm must choose a negative lenght even if there is a way to get to that point using an other route going back to where we come from. Now we can't let these answers become too big, since that would lead to infinite loops, so you can set an upper bound here:
    # If this equals k we will use the older method with the afforementioned error.
    solution_max_length::Integer
    ACOKSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, k, f, s) = new(ACOSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, (_, _) -> 1.0, (_, _) -> 1.0), k, f, s)
    ACOKSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s, k, f, s) = new(ACOSettings(α, β, n_a, ρ, ϵ, max_i, start_ph, e_f, c_s), k, f, s)
    ACOKSettings(acos, k, f, s) = new(acos, k, f, s)
end

mutable struct ACOInner
    graph
    n
    η
    τ
end


sample(weights) = findfirst(cumsum(weights) .> rand())


spread(inner::ACOInner) = inner.graph, inner.n, inner.η, inner.τ

# Calculates the probabilities of choosing edges to add to the solution.
function calculate_probabilities_old(inner::ACOInner, i, vars::ACOSettings, c)
    graph, n, η, τ = spread(inner)

    # graph.weights[i,j] * 
    p = [findfirst(x -> x == j, c) === nothing ? (τ[i, j]^vars.α * η[i, j]^vars.β) : 0 for j in 1:n]
    if maximum(p) == 0
        p[i] = 1
    end
    s_p = sum(p)

    p ./= s_p

    p
end

# Calculates the probabilities of choosing edges to add to the solution.
function calculate_probabilities(inner::ACOInner, i, vars::ACOSettings)
    _graph, n, η, τ = spread(inner)

    # graph.weights[i,j] * 
    p = [(τ[i, j]^vars.α * η[i, j]^vars.β) for j in 1:n]

    if maximum(p) == 0
        p[i] = 1
    end
    s_p = sum(p)

    p ./= s_p

    p
end

function generate_s(inner::ACOInner, vars::ACOKSettings, i)
    points = zeros(Int64, vars.k)
    n, _ = size(inner.graph)
    points[1] = i
    last = 1
    j = 2
    tries = 0
    canTry = ones(vars.k)
    while j <= vars.k
        points[j] = sample(calculate_probabilities(inner, points[last], vars.acos))
        if !(points[j] in points[1:(j-1)])
            j += 1
            last = j - 1
            tries = 0
        else
            tries += 1
            if tries == 50
                tries = 0
                canTry[last] = 0
                while (last > 0 && canTry[last] == 0)
                    last -= 1
                end
                if last == 0
                    points[j] = rand(1:n)
                    while !(points[j] in points[1:(j-1)])
                        points[j] = rand(1:n)
                    end
                    j += 1
                    last = j - 1
                end
            end
        end
    end

    points
end

# Constructs a new solution, the old way
function generate_s_old(inner::ACOInner, vars::ACOKSettings)
    i = rand(1:inner.n)
    points = zeros(Int64, vars.k)
    points[1] = i
    for i in 2:vars.k
        points[i] = sample(calculate_probabilities_old(inner, points[i-1], vars.acos, points))
        if points[i] == points[i-1]
            if vars.force_every_solution
                return generate_s_old(inner, vars)
            else
                return
            end
        end
    end

    points
end

function choose_iteration_best(inner::ACOInner, settings::ACOSettings, iterations)
    iterations = filter(x -> x !== nothing, iterations)
    points = Folds.map(x -> settings.eval_f(inner.graph, settings.compute_solution(inner.graph, x)), iterations)
    index = argmax(points)
    (iterations[index], points[index], length(iterations))
end


begin
    fst((a, _)) = a
    snd((_, b)) = b
end

function calculate_η_ij(graph, i, j)
    if graph[i, j] == 0
        return 0
    end

    count(graph[:, j] .!= 0)
end

function calculate_η(graph)
    n, _ = size(graph)

    m = minimum(graph)
    if m >= 0
        m = 0
    end
    η = [calculate_η_ij(graph, i, j) for i in 1:n, j in 1:n]

    η
end

function get_weight(g, (x, y))
    g.weights[x, y]
end

function calculateFitnessDense(graph, c)
    sum(map(x -> get_weight(graph, x), c))
end

function compute_solution(_graph, s)
    s
end

function ACOK(graph, vars::ACOKSettings, η, τ; logging=false)
    #Set parameters and initialize pheromone trails.
    n, _ = size(η)
    inner = ACOInner(graph, n, η, τ)

    logs = []

    @assert n >= vars.k
    @assert vars.k <= vars.solution_max_length
    sgb = [i for i in 1:n]
    sgb_val = -1000
    τ_max = vars.acos.starting_pheromone_ammount
    τ_min = 0

    fitnessEvals = 0

    # While termination condition not met
    while fitnessEvals < vars.acos.maxNumberOfEvals
        # Construct new solution s according to Eq. 2

        S = Folds.map(x -> generate_s(inner, vars, rand(1:inner.n)), zeros(vars.acos.number_of_ants))

        if length(filter(x -> x !== nothing, S)) > 0
            # Update iteration best
            (sib, sib_val, evalsUsed) = choose_iteration_best(inner, vars.acos, S)
            fitnessEvals += evalsUsed
            if sib_val > sgb_val
                sgb_val = sib_val
                sgb = sib

                # Compute pheromone trail limits
                τ_max = sgb_val / (1 - vars.acos.ρ)
                τ_min = vars.acos.ϵ * τ_max
            end
            # Update pheromone trails
            # TODO: test with matrix sum
            τ .*= (1 - vars.acos.ρ)
            for (a, b) in zip(sib, sib[2:end])
                τ[a, b] += sib_val
                τ[b, a] += sib_val

            end
        end
        τ = min.(τ, τ_max)
        τ = max.(τ, τ_min)

        if logging
            logRow = [fitnessEvals, sgb_val]
            append!(logRow, sort(sgb))
            push!(logs, logRow)
        end
    end

    vars.acos.compute_solution(inner.graph, sgb), τ, logs
end

function ACOK(graph, vars::ACOKSettings, η; logging=false)
    n, _ = size(η)
    τ = ones(n, n) .* vars.acos.starting_pheromone_ammount
    r, _, log = ACOK(graph, vars, η, τ; logging=logging)

    r, log
end


function ACOK_get_pheromone(graph, vars::ACOKSettings, η; logging=false)
    n, _ = size(η)
    τ = ones(n, n) .* vars.acos.starting_pheromone_ammount
    ACOK(graph, vars, η, τ; logging=logging)
end

function copy_replace_funcs(vars_base::ACOKSettings, eval_f, c_s)
    ACOKSettings(
        ACOSettings(
            vars_base.acos.α,
            vars_base.acos.β,
            vars_base.acos.number_of_ants,
            vars_base.acos.ρ,
            vars_base.acos.ϵ,
            vars_base.acos.maxNumberOfEvals,
            vars_base.acos.starting_pheromone_ammount,
            eval_f,
            c_s
        ),
        vars_base.k,
        vars_base.force_every_solution,
        vars_base.solution_max_length,
    )
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

function TrianglesACOK(graph, vars_base::ACOKSettings, τ; logging=false)
    η = calculate_η(graph)

    vars = copy_replace_funcs(vars_base, calculateFitnessDense, compute_solution)

    ACOK(graph, vars, η, τ; logging=logging)
end

function TrianglesACOK(graph, vars_base::ACOKSettings; logging=false)
    η = calculate_η(graph)

    vars = copy_replace_funcs(vars_base, calculateFitnessDense, compute_solution)

    ACOK(graph, vars, η; logging=logging)
end

function TrianglesACOKEta(graph, vars_base::ACOKSettings, η; logging=false)
    vars = copy_replace_funcs(vars_base, calculateFitnessDense, compute_solution)

    ACOK(graph, vars, η; logging=logging)
end

function TrianglesACOK_get_pheromone(graph, vars_base::ACOKSettings; logging=false)
    η = calculate_η(graph)

    vars = copy_replace_funcs(vars_base, calculateFitnessDense, compute_solution)

    ACOK_get_pheromone(graph, vars, η; logging=logging)
end


export TrianglesACOK, TrianglesACOK_get_pheromone, TrianglesACOKEta, ACOKSettings, ACOSettings
end