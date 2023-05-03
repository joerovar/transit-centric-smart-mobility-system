"""
Parameters for transit frequency setting problem

@author: Xiaotong Guo
"""

input_path = ""
output_path = ""

# Parameters
# Instance-based params
start_hour = 7; end_hour = 9;
articulated_bus_nm = 0;
standard_bus_seats = 37; standard_bus_capacity = 70;
articulated_bus_seats = 58; articulated_bus_capacity = 107;
minimum_time_interval_length = 5; # minutes
# Optimization-based params
γ = 1; threshold = 0.05; OPTIMALITY_GAP = 0.01;
num_threads = 18; TIME_LIMIT = 3600 * 3;
Γ_list = [1];

# Input
demand_df = CSV.read(input_path + "demand_direction1.csv", DataFrame);
demand_mean = CSV.read(input_path + "demand_mean_direction1.csv", DataFrame);
demand_std = CSV.read(input_path + "demand_std_direction1.csv", DataFrame);
route_pattern = CSV.read(input_path + "pattern_direction1_49.csv", DataFrame);
bus_supply = CSV.read(input_path + "supply_direction1.csv", DataFrame);
distance_running_time_df = CSV.read(input_path + "distance_running_time_10_2020.csv", DataFrame);

# Generate intermediate params
max_supply = sum(bus_supply[(bus_supply.bin .>= 12*start_hour) .& (bus_supply.bin .<= 12*end_hour-1), :].supply);

demand_df = demand_df[(demand_df.bin .>= 12*start_hour) .& (demand_df.bin .<= 12*end_hour-1), :]
demand_gd = groupby(demand_df, :date)
demand_mean = demand_mean[(demand_mean.bin .>= 12*start_hour) .& (demand_mean.bin .<= 12*end_hour-1), :]
demand_std = demand_std[(demand_std.bin .>= 12*start_hour) .& (demand_std.bin .<= 12*end_hour-1), :];

id_stop_dict = Dict()
stop_id_dict = Dict()
for ind in 1:size(route_pattern)[1]
    stop_id = route_pattern[ind,:stop_id]
    stop_id_dict[stop_id] = ind
    id_stop_dict[ind] = stop_id
end

pattern_ind = [stop_id_dict[i] for i in route_pattern.stop_id];

# Create demand scenarios
function generate_demand_scenario()
    demand_scenario = Dict()
    ind = 1
    for i in demand_gd
        demand_scenario[ind] = i
        ind += 1
    end
    return demand_scenario
end
demand_scenario = generate_demand_scenario()

travel_time_dict = Dict()
for i in eachrow(distance_running_time_df)
    segment = i.segment
    distance = i.stop_spacing
    travel_time = i.running_time
    stop_a = parse(Int64, split(segment, ", ")[1][2:end])
    stop_b = parse(Int64, split(segment, ", ")[2][1:end-1])
    if stop_a in keys(stop_id_dict) && stop_b in keys(stop_id_dict)
        travel_time_dict[(stop_id_dict[stop_a], stop_id_dict[stop_b])] = travel_time
    end
end

pattern_travel_time_dict = Dict()
travel_time_dict = Dict()
pattern_travel_time_dict["pattern1"] = travel_time_dict_pattern1

for i in 1:length(pattern_ind)-1
    segment = (pattern_ind[i], pattern_ind[i+1])
    travel_time_dict_pattern1[segment] = travel_time_dict[segment]
end

function create_parameters()
    parameters = (
        seed = 20230503,
        time_period_length = minimum_time_interval_length, # minutes
        first_bin = trunc(Int64, 12*start_hour),
        last_bin = trunc(Int64,12*end_hour - 1),
        total_time_intervals = trunc(Int64, 12*end_hour - 12*start_hour),
        vehicle_type = ["standard", "articulated"],
        vehicle_capacity = Dict("standard" => standard_bus_seats, "articulated" => standard_bus_capacity),
        maximum_capacity = Dict("standard" => articulated_bus_seats, "articulated" => articulated_bus_capacity),
        vehicle_budget = Dict("standard" => max_supply - articulated_bus_nm, "articulated" => articulated_bus_nm),
        total_budget = max_supply,
        demand_scenarios = collect(1:length(demand_gd)),
        stop_list = pattern_ind,
        segment_travel_time = pattern_travel_time_dict,
        pattern = pattern_ind,
        demand = demand_gd,
        demand_mean = demand_mean,
        demand_std = demand_std,
    )
    return parameters
end

params = create_parameters();

function demand_generation(p::NamedTuple)
    demand_gd = p.demand
    demand_mean = p.demand_mean
    demand_std = p.demand_std

    temp_df = demand_mean[(demand_mean.bin .>= p.first_bin),[:bin, :boarding_stop_id, :alighting_stop_id]]
    temp_df = temp_df[(temp_df.bin .<= p.last_bin), :]
    passenger_flow = []
    for i in eachrow(temp_df)
        push!(passenger_flow, (stop_id_dict[i.boarding_stop_id], stop_id_dict[i.alighting_stop_id], i.bin))
    end

    demand_stats = Dict()
    for i in passenger_flow
        μ = demand_mean[(demand_mean.bin .== i[3]) .& (demand_mean.boarding_stop_id .== id_stop_dict[i[1]]) .& (demand_mean.alighting_stop_id .== id_stop_dict[i[2]]), :demand][1]
        σ = demand_std[(demand_std.bin .== i[3]) .& (demand_std.boarding_stop_id .== id_stop_dict[i[1]]) .& (demand_std.alighting_stop_id .== id_stop_dict[i[2]]), :demand][1]
        demand_stats[i] = (round(μ, digits=2), round(σ, digits=2))
    end

    demand = Dict()
    for i in 1:length(demand_gd)
        demand[i] = Dict()
        for j in passenger_flow
            demand[i][j] = 0
        end
        for j in eachrow(demand_gd[i])
            demand[i][(stop_id_dict[j.boarding_stop_id], stop_id_dict[j.alighting_stop_id], j.bin)] = j.demand
        end
    end

    return demand, demand_stats, passenger_flow
end

function FrequencySettingData(p::NamedTuple)

    Δ = p.time_period_length
    T = p.total_time_intervals
    veh_B = p.vehicle_budget
    B = p.total_budget
    V = p.vehicle_type
    vehicle_capacity = p.vehicle_capacity
    max_capacity = p.maximum_capacity
    E = p.demand_scenarios
    S = p.stop_list
    P_list = ["pattern1"]
    P = Dict("pattern1" => p.pattern)
    u, u_stats, F = demand_generation(p)
    segment_travel_time = p.segment_travel_time

    prob = zeros(length(E))
    scenario_demand = zeros(length(E))
    for i in E
        scenario_demand[i] = sum(values(u[i]))
    end
    for i in E
        prob[i] = round(scenario_demand[i] / sum(scenario_demand), digits=2)
    end

    return FrequencySettingData(Δ, T, V, vehicle_capacity, max_capacity, veh_B, B, E, prob, S, P_list, P, u, u_stats, F, segment_travel_time, p.first_bin, p.last_bin)
end

data = FrequencySettingData(params);

total_demand = sum([i[1] for i in collect(values(data.u_stats))])
@show total_demand

function Base.show(io::IO, data::FrequencySettingData)
    @printf(io, "Frequency Setting data with %d stops, %d patterns and %d demand scenarios within %d time intervals",
            size(data.S, 1), length(collect(keys(data.P))), size(data.E, 1), data.T)
end
@show data
