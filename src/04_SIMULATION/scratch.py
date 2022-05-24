import matplotlib.pyplot as plt
import pandas as pd
from input import FOCUS_TRIPS

trajectories_inbound = pd.read_csv('in/vis/trajectories_inbound.csv')
trajectories_outbound = pd.read_csv('in/vis/trajectories_outbound.csv')
block_info = pd.read_csv('in/vis/block_info.csv')

block_info = block_info[['trip_id', 'block_id', 'route_type']]
trajectories_df = pd.concat([trajectories_outbound, trajectories_inbound], axis=0, ignore_index=True)

trajectories_df = trajectories_df.merge(block_info, on='trip_id')
focus_blocks = block_info[block_info['trip_id'].isin(FOCUS_TRIPS)]['block_id'].unique()
trajectories_df = trajectories_df[trajectories_df['block_id'].isin(focus_blocks)]
trajectories_df['stop_direction'] = trajectories_df['stop_id'].astype(str) + '_' + trajectories_df['route_type'].astype(str)
terminals = ['8613_1', '386_1', '15136_2', '386_2', '386_0', '8613_0']
trajectories_df = trajectories_df[trajectories_df['stop_direction'].isin(terminals)]
trajectories_df['date'] = trajectories_df['avl_arr_time'].str[8:10].astype(int)
trajectories_df = trajectories_df[['date','trip_id', 'block_id','stop_sequence','stop_id', 'avl_arr_sec', 'avl_dep_sec', 'schd_sec', 'route_type']]
trajectories_df = trajectories_df.sort_values(by=['date', 'schd_sec'])
trajectories_df['avl_arr_sec'] = trajectories_df['avl_arr_sec'] % 86400
trajectories_df['avl_dep_sec'] = trajectories_df['avl_dep_sec'] % 86400
trajectories_df.to_csv('in/vis/focus_block_trajectories.csv', index=False)
unique_dates = trajectories_df['date'].unique()
fig, axs = plt.subplots(ncols=3, sharex='all', sharey='all')
i = 0
for d in [3, 11, 16]:
    trajectories = trajectories_df[trajectories_df['date'] == d]

    trajectories_arr_sec = trajectories[['block_id', 'stop_id', 'avl_arr_sec']]
    trajectories_arr_sec = trajectories_arr_sec.rename(columns={'avl_arr_sec':'avl_sec'})
    trajectories_dep_sec = trajectories[['block_id', 'stop_id', 'avl_dep_sec']]
    trajectories_dep_sec = trajectories_dep_sec.rename(columns={'avl_dep_sec':'avl_sec'})
    trajectories_schd_sec = trajectories[['block_id', 'stop_id', 'schd_sec']]
    trajectories_schd_sec = trajectories_schd_sec.rename(columns={'schd_sec':'avl_sec'})
    trajectories_schd_sec['block_id'] = trajectories_schd_sec['block_id']*10

    trajectories_plot = pd.concat([trajectories_arr_sec, trajectories_dep_sec], ignore_index=True)
    trajectories_plot = trajectories_plot.sort_values(by='avl_sec')
    trajectories_plot = trajectories_plot.set_index('avl_sec')
    dict1 = {386: 0, 15136: 20, 8613: 67}
    trajectories_plot = trajectories_plot.replace({'stop_id': dict1})

    trajectories_plot.groupby(by='block_id')['stop_id'].plot(color='red', ax=axs[i])

    trajectories_plot2 = trajectories_schd_sec.sort_values(by='avl_sec')
    trajectories_plot2 = trajectories_plot2.set_index('avl_sec')
    dict1 = {386: 0, 15136: 20, 8613: 67}
    trajectories_plot2 = trajectories_plot2.replace({'stop_id': dict1})
    trajectories_plot2.groupby(by='block_id')['stop_id'].plot(color='silver', alpha=0.5, ax=axs[i])
    axs[i].set_xlim(26500, 34500)
    i += 1
plt.close()
fig, axs = plt.subplots(ncols=3, sharex='all', sharey='all')
trajectories_sim = pd.read_csv('out/trajectories/NC.csv')
terminals = [386, 8613]
i=0

for rep in [3, 11, 16]:
    trajectories = trajectories_sim[trajectories_sim['replication'] == rep]
    trajectories = trajectories[trajectories['stop_id'].isin(terminals)]

    trajectories_arr_sec = trajectories[['trip_id', 'stop_id', 'arr_sec']]
    trajectories_arr_sec = trajectories_arr_sec.rename(columns={'arr_sec':'avl_sec'})
    trajectories_dep_sec = trajectories[['trip_id', 'stop_id', 'dep_sec']]
    trajectories_dep_sec = trajectories_dep_sec.rename(columns={'dep_sec':'avl_sec'})

    trajectories_real = trajectories_df[trajectories_df['date'] == d]
    trajectories_real = trajectories_real[trajectories_real['trip_id'].isin(FOCUS_TRIPS)]
    trajectories_schd_sec = trajectories_real[['trip_id', 'stop_id', 'schd_sec']]
    trajectories_schd_sec = trajectories_schd_sec.rename(columns={'schd_sec':'avl_sec'})
    # trajectories_schd_sec['trip_id'] = trajectories_schd_sec['trip_id']*10

    trajectories_plot = pd.concat([trajectories_arr_sec, trajectories_dep_sec], ignore_index=True)
    trajectories_plot = trajectories_plot.sort_values(by='avl_sec')
    trajectories_plot = trajectories_plot.set_index('avl_sec')
    dict1 = {386: 0, 15136: 20, 8613: 67}
    trajectories_plot = trajectories_plot.replace({'stop_id': dict1})
    trajectories_plot.groupby(by='trip_id')['stop_id'].plot(color='red', ax=axs[i])

    trajectories_plot2 = trajectories_schd_sec.sort_values(by='avl_sec')
    trajectories_plot2 = trajectories_plot2.set_index('avl_sec')
    dict1 = {386: 0, 15136: 20, 8613: 67}
    trajectories_plot2 = trajectories_plot2.replace({'stop_id': dict1})
    trajectories_plot2.groupby(by='trip_id')['stop_id'].plot(color='silver', alpha=0.5, ax=axs[i])
    axs[i].set_xlim(26500, 34500)
    i += 1
# plt.show()


