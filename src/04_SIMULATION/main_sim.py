import numpy as np
import pandas as pd
from sim_env import run_base_detailed, run_base_control_detailed
from file_paths import *
from post_process import load, plot_2_var_whisker, plot_sensitivity_whisker_compliance, plot_3_var_whisker, \
    plot_sensitivity_whisker_run_t
from output import PostProcessor
import matplotlib.pyplot as plt
import seaborn as sns


def run_benchmark(base=True, base_control=True, control_strength=0.7, hold_adj_factor=0.0, tt_factor=1.0):
    if base:
        run_base_detailed(replications=N_REPLICATIONS, save_results=True)
    if base_control:
        run_base_control_detailed(replications=N_REPLICATIONS, save_results=True, control_strength=control_strength,
                                  hold_adj_factor=hold_adj_factor, tt_factor=tt_factor)
    return


def validate_nc():
    prc = PostProcessor([path_tr_nc_b2],
                        [path_p_nc_b2], ['NC'], N_REPLICATIONS,
                        'out/compare/validate/')
    prc.pax_profile_base()
    prc.write_trajectories(only_nc=True)
    results = {}
    results.update(prc.headway(save_nc=True))
    results.update(prc.trip_times(keep_nc=True))
    cv_hw_observed = load('in/xtr/cv_hw_outbound.pkl')
    # cv_hw_observed = load('in/xtr/cv_hw_out2.pkl')
    cv_hw_sim = load('out/compare/validate/cv_hw_sim.pkl')
    plt.plot(cv_hw_observed, label='observed')
    plt.plot(cv_hw_sim, label='simulated')
    plt.xlabel('stop')
    plt.ylabel('headway coefficient of variation')
    plt.legend()
    plt.savefig('out/compare/validate/cv_hw.png')
    plt.close()
    trip_t_obs = load('in/xtr/trip_t_outbound.pkl')
    t_out = load('out/compare/validate/trip_t_sim.pkl')
    fig, ax = plt.subplots(1, sharex='all')
    ax.hist([[i / 60 for i in trip_t_obs], [t / 60 for t in t_out]], color=['black', 'gray'], alpha=0.5,
            density=True, label=['observed', 'simulated'], bins=12, ec='black')
    ax.set_xlabel('trip time (min)')
    # ax.set_yticks(np.arange(0.0, 0.16, 0.04))
    # ax.set_yticklabels([str(f) + '%' for f in range(0, 16, 4)])
    ax.set_ylabel('frequency (%)')
    # plt.xlim(60, 81.0)
    fig.legend()
    plt.tight_layout()
    plt.savefig('out/compare/validate/trip_t.png')
    plt.close()
    return


def weight_comparison(compute_rbt=False):
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


def fancy_plots():
    wt_all_set2 = load(path_dir_s1 + 'wt_numer.pkl')
    nr_replications = 40
    nr_methods = 3
    nr_scenarios = 3
    idx = [0, 2, 4, 5, 6, 7, 8, 10, 12]
    wt = list(np.array(wt_all_set2)[idx].flatten())

    method = (['EH'] * nr_replications + ['RL-LA'] * nr_replications + ['RL-HA'] * nr_replications) * nr_scenarios
    tt_var = ['low'] * nr_replications * nr_methods + ['base'] * nr_replications * nr_methods + ['high'] * nr_replications * nr_methods

    df_dict = {'tt_var': tt_var, 'method': method, 'wt': wt}
    df = pd.DataFrame(df_dict)
    sns.set(style='darkgrid')
    sns.boxplot(x='tt_var', y='wt', hue='method', data=df, showfliers= False)
    plt.ylabel('mean wait time (min)')
    plt.xlabel('run time variability')
    plt.savefig('out/compare/sensitivity run times/wt_fancy.png')
    plt.close()

    wt_set = load(path_dir_s2 + 'wt_numer.pkl')

    nr_replications = 40
    nr_methods = 3
    nr_scenarios = 3
    idx = [0, 1, 2, 3, 5, 6, 8, 9, 11]
    wt = list(np.array(wt_set)[idx].flatten())

    method = (['EH'] * nr_replications + ['RL-LA'] * nr_replications + ['RL-HA'] * nr_replications) * nr_scenarios
    compliance = [100] * nr_replications * nr_methods + [80] * nr_replications * nr_methods + [60] * nr_replications * nr_methods

    df_dict = {'compliance': compliance, 'method': method, 'wt': wt}
    df = pd.DataFrame(df_dict)
    sns.set(style='darkgrid')
    sns.boxplot(x='compliance', y='wt', hue='method', data=df, showfliers= False)
    plt.legend('')
    plt.xlabel('degree of compliance (%)')
    plt.ylabel('mean wait time (min)')
    plt.savefig('out/compare/sensitivity compliance/wt_fancy.png')
    plt.close()

    return


N_REPLICATIONS = 40

run_base_detailed(replications=10, save_results=True)
# validate_nc()
# run_base_control_detailed(replications=40, save_results=True)
# prc = PostProcessor([path_tr_nc_b2, path_tr_eh_b2],
#                     [path_p_nc_b2, path_p_eh_b2],
#                     cp_tags=['NC', 'EH'], path_dir='out/compare/ncvseh/', nr_reps=N_REPLICATIONS)
# prc.headway(plot_cv=True)
# results = {}
# results.update(prc.load_profile(plot_single=True))
# prc.pax_profile_base()
# # run_benchmark(base=False, base_control=True, control_strength=0.75, tt_factor=0.8)
# run_benchmark(base=False, base_control=True, control_strength=0.75, tt_factor=1.2)
# weight_comparison(compute_rbt=True)
# benchmark_comparison(compute_rbt=False)
# sensitivity_run_t(compute_rbt=True)

# sensitivity_compliance(compute_rbt=True)
