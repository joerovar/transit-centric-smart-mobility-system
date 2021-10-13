import simulation_run
import output
import time
st = time.time()


# running
simulation_run.run(save=False)

# outputs
# output.get_results()

print("ran in %.2f seconds" % (time.time()-st))
