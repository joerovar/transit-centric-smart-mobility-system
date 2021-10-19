import simulation_run
import output
import time
st = time.time()


# running
# simulation_run.run_base(save=False)
simulation_run.run_base_drl(save=True)

# outputs
output.get_results()
output.get_rl_results()

print("ran in %.2f seconds" % (time.time()-st))
