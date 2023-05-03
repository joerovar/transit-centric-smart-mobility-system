# transit-centric-smart-mobility-system
## Robust Optimization Module
### Computation Environment Requirements
In order to run the RO module, a Julia version v1.1 need to be installed with the following packages:
- Atom v0.12.30
- BSON v0.3.3
- CSV v0.8.2
- DataFrames v0.21.8
- Distributions v0.24.10
- Gurobi v0.8.1
- IJulia v1.23.1
- JSON v0.21.1
- JuMP v0.18.6
- JuMPeR v0.6.0
- Juno v0.8.4
- NPZ v0.4.1
- ProgressBars v1.3.0
- PyCall v1.92.3
- StatsBase v0.33.8

### Module Input
- `demand_df`: full demand data
- `demand_mean`: mean of demand data
- `demand_std`: standard deviation of demand data
- `route_pattern`: route patterns
- `bus_supply`: actual bus operation schedules
- `distance_running_time_df`: running time between segments

### Module Parameters
- `start_hour` (0 to 24)
- `end_hour` (0 to 24, larger than start hour)
- `articulated_bus_nm`: number of articulated buses
- `standard_bus_seats`, `standard_bus_capacity`
- `articulated_bus_seats`, `articulated_bus_capacity`
- `minimum_time_interval_length` (minutes)
- `γ`: weight between wait time and in-vehicle travel time
- `threshold`: stop threshold for Gurobi
- `OPTIMALITY_GAP`: stop threshold for Gurobi
- `num_threads`: number of threads used in the Gurobi
- `TIME_LIMIT`: maximum computation time for Gurobi
- `Γ_list`: list of Gamma which defines the size of uncertainty set

### Module Output
- optimized schedule for baseline and robust optimization in `npz` format
