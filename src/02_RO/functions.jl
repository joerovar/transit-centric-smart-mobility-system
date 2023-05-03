"""
Functions for transit frequency setting problem

@author: Xiaotong Guo
"""


function init_to_board_travel_time(data::FrequencySettingData, F_new)
    init_to_board_travel_time_dict = Dict()
    for p in 1:length(data.P_list), f in 1:length(F_new)
        (o, d, t) = F_new[f]
        travel_time = 0
        for i in 1:o-1
            segment = (i, i+1)
            travel_time += data.segment_travel_time[data.P_list[p]][segment]
        end
        travel_time = round(travel_time, digits=3)
        init_to_board_travel_time_dict[(f,p)]  = (travel_time, Int(floor(travel_time / data.Δ)))
    end
    return init_to_board_travel_time_dict
end


function loading_index_calculation(data::FrequencySettingData, init_to_board_tt::Dict, F_new)
    L_index_dict = Dict((s, τ, p) => [] for s in data.S, τ in data.first_bin:data.last_bin, p in 1:length(data.P_list))
    for f in 1:length(F_new)
        (o,d,t) = F_new[f]
        for s in o:d-1
            for p in 1:length(data.P_list)
                pattern_stop_list = data.P[data.P_list[p]]
                if o in pattern_stop_list && d in pattern_stop_list
                    for τ in max(data.first_bin, t - init_to_board_tt[(f, p)][2]):data.last_bin
                        push!(L_index_dict[(s,τ,p)], f)
                    end
                end
            end
        end
    end
    return L_index_dict
end


function passenger_flow_to_pattern_func(data::FrequencySettingData, F_new)
    passenger_flow_to_pattern_dict = Dict(i => [] for i in 1:length(F_new))
    for f in 1:length(F_new)
        (o,d,t) = F_new[f]
        for p in 1:length(data.P_list)
            pattern_stop_list = data.P[data.P_list[p]]
            if o in pattern_stop_list && d in pattern_stop_list
                push!(passenger_flow_to_pattern_dict[f], p)
            end
        end
    end
    return passenger_flow_to_pattern_dict
end


function in_vehicle_travel_time(data::FrequencySettingData, F_new)

    in_vehicle_travel_time_dict = Dict()
    infeasible_tt = 1e5

    for f in 1:length(F_new)
        (o,d,t) = F_new[f]
        for p in 1:length(data.P_list)
            pattern_stop_list = data.P[data.P_list[p]]
            if o in pattern_stop_list && d in pattern_stop_list
                feasible_tt = 0
                o_ind = findall(x->x==o, pattern_stop_list)[1]
                d_ind = findall(x->x==d, pattern_stop_list)[1]
                for i in o_ind:d_ind-1
                    segment = (i, i+1)
                    feasible_tt += data.segment_travel_time[data.P_list[p]][segment]
                end
                feasible_tt = round(feasible_tt, digits=3)
                in_vehicle_travel_time_dict[(f, p)] = feasible_tt
            else
                in_vehicle_travel_time_dict[(f, p)] = infeasible_tt
            end
        end
    end

    return in_vehicle_travel_time_dict
end
