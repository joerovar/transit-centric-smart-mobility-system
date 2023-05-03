"""
Robust Optimization for transit schedule optimizaiton problem

@author: Xiaotong Guo
"""

using CSV, DataFrames, JuMP, Gurobi
using NPZ, Statistics, LinearAlgebra
using ProgressBars, Random
using JuMPeR, Distributions, Printf, StatsBase, BSON
using Dates, JSON

cd("/Users/xiaotong/Documents/GitHub/transit-centric-smart-mobility-system/")
include("/Users/xiaotong/Documents/GitHub/transit-centric-smart-mobility-system/src/02_RO/structures.jl")
include("/Users/xiaotong/Documents/GitHub/transit-centric-smart-mobility-system/src/02_RO/functions.jl")
include("/Users/xiaotong/Documents/GitHub/transit-centric-smart-mobility-system/src/02_RO/parameters.jl")
include("/Users/xiaotong/Documents/GitHub/transit-centric-smart-mobility-system/src/02_RO/robust_optimization_functions.jl")

println("Running Robust Optimization Module...")

# Baseline optimization with one day demand scenario
demand_scenario = Dict()
for j in keys(data.u[1])
    if data.u[1][j] > 0
        demand_scenario[j] = data.u[1][j]
    end
end

@show sum(values(demand_scenario))
opt_val, opt_x, opt_y, opt_λ, opt_L = baseline_opt(data, γ, demand_scenario);
npzwrite(output_path + "baseline_opt_schedules.npz", opt_x[:,:,:])

# Robust optimization with Transit Downsizing
for Γ in tqdm(Γ_list)
    adjusted_data = deepcopy(data)
    new_u_stats = Dict()
    for i in keys(data.u_stats)
        if data.u_stats[i][1] > threshold
            new_u_stats[i] = data.u_stats[i]
        end
    end
    new_F = collect(keys(new_u_stats))
    adjusted_data.u_stats = new_u_stats
    adjusted_data.F = new_F

    x_RO = robust_opt(adjusted_data, Γ, γ)
    npzwrite(output_path + "RO_schedules_gamma_"*string(Γ)*".npz", x_RO[:,:,:]);
end

println("Robust Optimization Module is successfully run and the optimal schedule is generated")
