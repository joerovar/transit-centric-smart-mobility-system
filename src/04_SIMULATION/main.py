import simul_run
import output
import time
st = time.time()

# simul_run.run(save=True)
# output.write_results()
# output.plot_results()
output.combine_episodes()
print("----%.2f----" % (time.time()-st))
