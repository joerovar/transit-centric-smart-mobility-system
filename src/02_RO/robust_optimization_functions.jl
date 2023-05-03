"""
Robust Optimization for transit schedule optimizaiton problem

@author: Xiaotong Guo
"""

using CSV, DataFrames, JuMP, Gurobi
using NPZ, Statistics, LinearAlgebra
using ProgressBars, Random
using JuMPeR, Distributions, Printf, StatsBase, BSON
using Dates, JSON


function baseline_opt(data::FrequencySettingData, Gamma, demand_scenario; fixed_x = nothing, verbose::Bool = true)
    model = Model(solver=GurobiSolver(TimeLimit=TIME_LIMIT, MIPGap=OPTIMALITY_GAP, OutputFlag=ifelse(verbose, 1, 0)))

    F_new = collect(keys(demand_scenario))
    P_n = length(data.P_list)
    V_n = length(data.V)
    F_n = length(F_new)
    T1, T2 = data.first_bin, data.last_bin
    init_to_board_tt = init_to_board_travel_time(data, F_new)
    L_ind_dict = loading_index_calculation(data, init_to_board_tt, F_new) # key: (s, τ, p)
    pax_flow_to_pattern = passenger_flow_to_pattern_func(data, F_new)
    in_vehicle_tt = in_vehicle_travel_time(data, F_new)
    γ = Gamma
    M = 1e5

    # Decision variable x and the feasible region
    @variable(model, x[p=1:P_n, v=1:V_n, t=T1:T2], Bin)
    # @constraint(model, [t=T1:T2], sum(x[p,v,t] for p in 1:P_n, v in 1:V_n) <= 1)
    if !isnothing(fixed_x)
        @constraint(model, [p=1:P_n, v=1:V_n, t=T1:T2], x[p,v,t] == fixed_x[p,v,t])
    end
    @constraint(model, [t=T1:T2, p=1:P_n], sum(x[p,v,t] for v in 1:V_n) <= 1)

    @constraint(model, [v=1:V_n], sum(x[p,v,t] for p in 1:P_n, t in T1:T2) <= data.vehicle_B[data.V[v]])
    @constraint(model, sum(x[p,v,t] for p in 1:P_n, t in T1:T2, v in 1:V_n) <= data.total_B)

    # Decision variable lambda for boarding
    @variable(model, λ[f=1:F_n, p=1:P_n, v=1:V_n, τ=max(T1, (F_new[f][3] - init_to_board_tt[(f, p)][2])):T2] >= 0)
    @variable(model, L[τ=T1:T2, p=1:P_n, v=1:V_n, s in data.P[data.P_list[p]]] >= 0)

    # Problem Constraints (3)
    @constraint(model, [τ=T1:T2, p=1:P_n, v=1:V_n, s in data.P[data.P_list[p]]],
            L[τ,p,v,s] == sum(λ[f, p, v, τ] for f in L_ind_dict[(s, τ, p)]))

    # Problem Constraints (4c)
    @constraint(model, [τ=T1:T2, p=1:P_n, v=1:V_n, s in data.P[data.P_list[p]]],
        L[τ,p,v,s] <= x[p,v,τ] * data.max_vehicle_capacity[data.V[v]])

    # Add variable for unsatisfied demand
    @variable(model, y[f=1:F_n]>=0)

    # Problem Constraints (4b)
    for f in 1:F_n
        @constraint(model, sum(λ[f, p, v, τ] for v=1:V_n, p in pax_flow_to_pattern[f],
            τ=max(T1, (F_new[f][3] - init_to_board_tt[(f, p)][2])):T2) == demand_scenario[F_new[f]]- y[f])
    end

    # Objectives
    @objective(model, Min, sum(data.Δ * (τ + init_to_board_tt[(f, p)][2] - F_new[f][3])*λ[f, p, v, τ] for f=1:F_n, p=1:P_n, v=1:V_n, τ=max(T1, (F_new[f][3] - init_to_board_tt[(f, p)][2])):T2)
        + γ * sum(in_vehicle_tt[f,p] * λ[f, p, v, τ] for f=1:F_n, p=1:P_n, v=1:V_n, τ=max(T1, (F_new[f][3] - init_to_board_tt[(f, p)][2])):T2)
        + M * sum(y[f] for f=1:F_n))

    solve(model)

    return getobjectivevalue(model), getvalue(x), getvalue(y), getvalue(λ), getvalue(L)
end


function robust_opt(data, Γ, Gamma; verbose=true)
    robust_model = RobustModel(solver=GurobiSolver(TimeLimit=TIME_LIMIT,
            MIPGap=OPTIMALITY_GAP, OutputFlag=ifelse(verbose, 1, 0),
            NodefileStart=0.5, Threads=num_threads))

    F_new = data.F
    P_n = length(data.P_list)
    V_n = length(data.V)
    F_n = length(F_new)
    T1, T2 = data.first_bin, data.last_bin
    init_to_board_tt = init_to_board_travel_time(data, F_new)
    L_ind_dict = loading_index_calculation(data, init_to_board_tt, F_new) # key: (s, τ, p)
    pax_flow_to_pattern = passenger_flow_to_pattern_func(data, F_new)
    in_vehicle_tt = in_vehicle_travel_time(data, F_new)
    γ = Gamma
    M = 1e5

    @variable(robust_model, x[p=1:P_n, v=1:V_n, t=T1:T2], Bin)
    @constraint(robust_model, [t=T1:T2, p=1:P_n], sum(x[p,v,t] for v in 1:V_n) <= 1)
    @constraint(robust_model, [v=1:V_n], sum(x[p,v,t] for p in 1:P_n, t in T1:T2) <= data.vehicle_B[data.V[v]])
    @constraint(robust_model, sum(x[p,v,t] for p in 1:P_n, t in T1:T2, v in 1:V_n) <= data.total_B)

    # Decision variable lambda for boarding
    @variable(robust_model, λ[f=1:F_n, p=1:P_n, v=1:V_n, τ=max(T1, (F_new[f][3] - init_to_board_tt[(f, p)][2])):T2] >= 0)
    @variable(robust_model, L[τ=T1:T2, p=1:P_n, v=1:V_n, s in data.P[data.P_list[p]]] >= 0)
    @constraint(robust_model, [τ=T1:T2, p=1:P_n, v=1:V_n, s in data.P[data.P_list[p]]],
        L[τ,p,v,s] == sum(λ[f, p, v, τ] for f in L_ind_dict[(s, τ, p)]))

    @constraint(robust_model, [τ=T1:T2, p=1:P_n, v=1:V_n, s in data.P[data.P_list[p]]],
    L[τ,p,v,s] <= x[p,v,τ] * data.max_vehicle_capacity[data.V[v]])

    # Uncertain Parameter & sets
    @uncertain(robust_model, 𝜁[f=1:F_n])
    @constraint(robust_model, norm(𝜁, 1) <= Γ)
    @constraint(robust_model, norm(𝜁, Inf) <= 1)

    # Add variable for unsatisfied demand
    @variable(robust_model, y[f=1:F_n]>=0)
    for f in 1:F_n
        @constraint(robust_model, sum(λ[f, p, v, τ] for v=1:V_n, p in pax_flow_to_pattern[f],
                    τ=max(T1, (F_new[f][3] - init_to_board_tt[(f, p)][2])):T2) == data.u_stats[F_new[f]][1] + data.u_stats[F_new[f]][2] * 𝜁[f] - y[f] )
    end

    # Objectives
    @objective(robust_model, Min, sum(data.Δ * (τ + init_to_board_tt[(f, p)][2] - F_new[f][3])*λ[f, p, v, τ] for f=1:F_n, p=1:P_n, v=1:V_n, τ=max(T1, (F_new[f][3] - init_to_board_tt[(f, p)][2])):T2)
        + γ * sum(in_vehicle_tt[f,p] * λ[f, p, v, τ] for f=1:F_n, p=1:P_n, v=1:V_n, τ=max(T1, (F_new[f][3] - init_to_board_tt[(f, p)][2])):T2)
        + M * sum(y[f] for f=1:F_n))

    solve(robust_model)

    return getvalue(x)
end
