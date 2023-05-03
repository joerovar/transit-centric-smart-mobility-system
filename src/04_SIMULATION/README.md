# flex-sim
This bus route simulator was built with the following principles: 
- compatibility with transit agency data feeds (GTFS, AVL, APC, ODX)
- individual passenger journey modeling
- modeling of uncertainty in service delivered (driver absenteeism)
- operational analysis of control strategies (holding at stops, expressing, departure readjustments)
- output of performance metrics such as service reliability, crowding, on-time performance
- extendable to multiple inter-dependent bus lines and strategies such as inter-lining

Transit data must be in directory {PLACEHOLDER}:
- GTFS zip file
- AVL feed containing stop arrival information, including boardings/alightings
- ODX (insert reference) with inferred destination stop IDs for fare card tap-ins (used in WMATA/CTA/MBTA/TfL)

Dependencies are listed in requirements.txt

Steps:
1. Run {PLACEHOLDER FUNCTION} in {PLACEHOLDER} to load simulation data
2. Specify fixed inputs {PLACEHOLDER FOR FIXED INPUTS}
3. Run {PLACEHOLDER FUNCTION} in main.py
4. Run evaluator function {PLACEHOLDER FUNCTION} in 

Secondary applications (future): 
Based on fixed route layout (e.g., GTFS stop IDs and schedules), generate a flexible route system where passengers need to request a trip.
