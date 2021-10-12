import simul_run
import output
import time
st = time.time()


# running
simul_run.run(save=True)

# outputs
output.get_results()

print("ran in %.2f seconds" % (time.time()-st))
