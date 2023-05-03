"""
Data structure for transit frequency setting problem

@author: Xiaotong Guo
"""


mutable struct FrequencySettingData
    "Time interval length"
    Δ::Int64
    "Number of time intervals"
    T::Int64
    "Vehicle types"
    V::Vector
    "Vehicle capacities"
    vehicle_capacity::Dict
    "Maximum vehicle capacities"
    max_vehicle_capacity::Dict
    "The total budget for deploying transit services/vehicles of a certain vehicle type"
    vehicle_B::Dict
    "Total budget for operations"
    total_B::Int64
    "Demand scenarios"
    E::Vector
    "Probability of each scenarios"
    prob::Vector
    "Stops"
    S::Vector
    "Patterns list"
    P_list::Vector
    "Patterns"
    P::Dict
    "Passenger flow demand for each scenarios"
    u::Dict
    "Passenger flow demand average and standard deviation"
    u_stats::Dict
    "Set of passenger flow"
    F::Vector
    "Segment travel time"
    segment_travel_time::Dict
    "First time bin"
    first_bin::Int64
    "Last time bin"
    last_bin::Int64
end
