import numpy as np
import pandas as pd
from sim_env import run_base_detailed, run_base_control_detailed
from file_paths import *
from post_process import load, plot_2_var_whisker, plot_sensitivity_whisker_compliance, plot_3_var_whisker, \
    plot_sensitivity_whisker_run_t
from output import PostProcessor


def run_benchmark(base=True, base_control=True, control_strength=0.7, hold_adj_factor=0.0, tt_factor=1.0):
    # RUN BENCHMARK
    if base:
        run_base_detailed(replications=N_REPLICATIONS, save_results=True)
    if base_control:
        run_base_control_detailed(replications=N_REPLICATIONS, save_results=True, control_strength=control_strength,
                                  hold_adj_factor=hold_adj_factor, tt_factor=tt_factor)
    return


def validate_non_rl(compute_rbt=False):
    # TEST NON RL
    prc = PostProcessor([path_tr_nc_b, path_tr_eh_b],
                        [path_p_nc_b, path_p_eh_b], ['NC', 'EH'], N_REPLICATIONS,
                        'out/compare/test/')
    prc.pax_profile_base()

    results = {}
    results.update(prc.pax_times_fast(include_rbt=compute_rbt))
    results.update(prc.headway())
    results.update(prc.trip_times(keep_nc=True))
    results_df = pd.DataFrame(results, columns=list(results.keys()))
    results_df.to_csv('out/compare/test/numer_results.csv', index=False)

    return


def weight_comparison(compute_rbt=False):
    # WEIGHTS COMPARISON
    prc_w = PostProcessor([path_tr_ddqn_ha3, path_tr_ddqn_ha5, path_tr_ddqn_ha7, path_tr_ddqn_ha9, path_tr_ddqn_ha11],
                          [path_p_ddqn_ha3, path_p_ddqn_ha5, path_p_ddqn_ha7, path_p_ddqn_ha9, path_p_ddqn_ha11],
                          tags_w, N_REPLICATIONS, path_dir_w)
    results_w = {}
    results_w.update(prc_w.pax_times_fast(include_rbt=compute_rbt))

    prc_w.write_trajectories()
    results_w.update(prc_w.control_actions())
    results_w.update(prc_w.trip_times())
    results_w.update(prc_w.headway())
    rbt_od_set = load(path_dir_w + 'rbt_numer.pkl')
    results_w.update({'rbt_mean': [np.around(np.mean(rbt), decimals=2) for rbt in rbt_od_set],
                      'rbt_median': [np.around(np.median(rbt), decimals=2) for rbt in rbt_od_set]})
    results_df = pd.DataFrame(results_w, columns=list(results_w.keys()))
    results_df.to_csv(path_dir_w + 'numer_results.csv', index=False)
    wt_all_set = load(path_dir_w + 'wt_numer.pkl')
    trip_t_all_set = load(path_dir_w + 'all_trip_t.pkl')
    plot_3_var_whisker(rbt_od_set, wt_all_set, trip_t_all_set,tags_w, path_dir_w + 'pax_times.png', 'reliability buffer time (min)',
                       'avg pax wait time (min)', 'trip time (min)', x_label=r'$W_{wait}$')
    return


def benchmark_comparison(compute_rbt=False):
    prc = PostProcessor([path_tr_nc_b, path_tr_eh_b, path_tr_ddqn_la_b,
                         path_tr_ddqn_ha_b],
                        [path_p_nc_b, path_p_eh_b, path_p_ddqn_la_b,
                         path_p_ddqn_ha_b], tags_b, N_REPLICATIONS,
                        path_dir_b)

    prc.pax_profile_base()
    prc.sample_trajectories()
    results = {}
    results.update(prc.pax_times_fast(include_rbt=compute_rbt))

    rbt_od_set = load(path_dir_b + 'rbt_numer.pkl')
    results.update({'rbt_mean': [np.around(np.mean(rbt), decimals=2) for rbt in rbt_od_set],
                    'rbt_median': [np.around(np.median(rbt), decimals=2) for rbt in rbt_od_set]})
    results.update(prc.headway(plot_bars=True))
    results.update(prc.load_profile(plot_grid=True))
    results.update(prc.trip_times(keep_nc=True, plot=True))
    prc.write_trajectories()
    results.update(prc.control_actions())
    results_df = pd.DataFrame(results, columns=list(results.keys()))
    results_df.to_csv(path_dir_b + 'numer_results.csv', index=False)

    wt_all_set = load(path_dir_b + 'wt_numer.pkl')
    plot_2_var_whisker(rbt_od_set, wt_all_set, tags_b, path_dir_b + 'pax_times.png', 'reliability buffer time (min)',
                       'avg pax wait time (min)')
    return


def sensitivity_run_t(compute_rbt=False):
    prc = PostProcessor(
        [path_tr_eh_low_s1, path_tr_ddqn_la_low_s1_nr, path_tr_ddqn_la_low_s1, path_tr_ddqn_ha_low_s1_nr,
         path_tr_ddqn_ha_low_s1,
         path_tr_eh_base_s1, path_tr_ddqn_la_base_s1, path_tr_ddqn_ha_base_s1,
         path_tr_eh_high_s1, path_tr_ddqn_la_high_s1_nr, path_tr_ddqn_la_high_s1, path_tr_ddqn_ha_high_s1_nr,
         path_tr_ddqn_ha_high_s1],
        [path_p_eh_low_s1, path_p_ddqn_ha_low_s1_nr, path_p_ddqn_la_low_s1, path_p_ddqn_ha_low_s1_nr,
         path_p_ddqn_ha_low_s1,
         path_p_eh_base_s1, path_p_ddqn_la_base_s1, path_p_ddqn_ha_base_s1,
         path_p_eh_high_s1, path_p_ddqn_la_high_s1_nr, path_p_ddqn_la_high_s1, path_p_ddqn_ha_high_s1_nr,
         path_p_ddqn_ha_high_s1], tags_s1, N_REPLICATIONS, path_dir_s1)
    results = {}
    results.update(prc.pax_times_fast(include_rbt=compute_rbt))

    rbt_od_set = load(path_dir_s1 + 'rbt_numer.pkl')
    wt_all_set = load(path_dir_s1 + 'wt_numer.pkl')
    plot_sensitivity_whisker_run_t(rbt_od_set, wt_all_set, ['EH', 'DDQN-LA (NR)' ,'DDQN-LA (R)', 'DDQN-HA (NR)', 'DDQN-HA (R)'],
                             ['cv: -20%', 'cv: base', 'cv: +20%'], ['EH', 'DDQN-LA', 'DDQN-HA'],
                             'reliability buffer time (min)', 'avg pax wait time (min)', path_dir_s1 + 'pax_times.png')
    results.update({'rbt_od': [np.around(np.mean(rbt), decimals=2) for rbt in rbt_od_set]})
    results.update(prc.headway())
    results_df = pd.DataFrame(results, columns=list(results.keys()))
    results_df.to_csv(path_dir_s1 + 'numer_results.csv', index=False)
    return


def sensitivity_compliance(compute_rbt=False):
    prc = PostProcessor([path_tr_eh_base_s2, path_tr_ddqn_la_base_s2, path_tr_ddqn_ha_base_s2,
                         path_tr_eh_80_s2, path_tr_ddqn_la_80_s2_nr, path_tr_ddqn_la_80_s2, path_tr_ddqn_ha_80_s2_nr,
                         path_tr_ddqn_ha_80_s2, path_tr_eh_60_s2, path_tr_ddqn_la_60_s2_nr, path_tr_ddqn_la_60_s2,
                         path_tr_ddqn_ha_60_s2_nr, path_tr_ddqn_ha_60_s2],
                        [path_p_eh_base_s2, path_p_ddqn_la_base_s2, path_p_ddqn_ha_base_s2,
                         path_p_eh_80_s2, path_p_ddqn_la_80_s2_nr, path_p_ddqn_la_80_s2, path_p_ddqn_ha_80_s2_nr,
                         path_p_ddqn_ha_80_s2, path_p_eh_60_s2, path_p_ddqn_la_60_s2_nr, path_p_ddqn_la_60_s2,
                         path_p_ddqn_ha_60_s2_nr, path_p_ddqn_ha_60_s2], tags_s2, N_REPLICATIONS,
                        path_dir_s2)
    results = {}
    results.update(prc.pax_times_fast(include_rbt=compute_rbt))
    rbt_od_set = load(path_dir_s2 + 'rbt_numer.pkl')
    wt_all_set = load(path_dir_s2 + 'wt_numer.pkl')
    plot_sensitivity_whisker_compliance(rbt_od_set, wt_all_set,
                                        ['EH', 'DDQN-LA (NR)', 'DDQN-LA (R)', 'DDQN-HA (NR)', 'DDQN-HA (R)'],
                                        ['base', '0.8', '0.6'], ['EH', 'DDQN-LA', 'DDQN-HA'],
                                        'reliability buffer time (min)', 'avg pax wait time (min)',
                                        path_dir_s2 + 'pax_times.png')
    results.update({'rbt_mean': [np.around(np.mean(rbt), decimals=2) for rbt in rbt_od_set],
                    'rbt_median': [np.around(np.median(rbt), decimals=2) for rbt in rbt_od_set]})
    results.update(prc.headway())
    results_df = pd.DataFrame(results, columns=list(results.keys()))
    results_df.to_csv(path_dir_s2 + 'numer_results.csv', index=False)
    return


N_REPLICATIONS = 40
# run_base_detailed(replications=40, save_results=True)

# prc = PostProcessor([path_tr_nc_b2, path_tr_nc_b],
#                     [path_p_nc_b2, path_p_nc_b],
#                     cp_tags=['new demand', 'old demand'], path_dir='out/compare/', nr_reps=N_REPLICATIONS)
# results = {}
# results.update(prc.load_profile(plot_single=True))
# prc.pax_profile_base()
# # run_benchmark(base=False, base_control=True, control_strength=0.75, tt_factor=0.8)
# run_benchmark(base=False, base_control=True, control_strength=0.75, tt_factor=1.2)
# weight_comparison(compute_rbt=True)
# benchmark_comparison(compute_rbt=False)
# sensitivity_run_t(compute_rbt=True)
# validate_non_rl(compute_rbt=False)
# sensitivity_compliance(compute_rbt=True)
