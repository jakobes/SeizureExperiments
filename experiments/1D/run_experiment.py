import resource
import warnings
import time

from multiprocessing import Pool
from itertools import product
from pathlib import Path

from single_experiment import (
    run_ML_experiment,
)

warnings.simplefilter("ignore", UserWarning)


if __name__ == "__main__":
    conductivity_list = (2, 0.5)
    KL_list = (0.5, 0.5 / 2, 0.5 / 4)
    parameter_list = product(conductivity_list, KL_list)

    def experiment(params, N=300, dt=0.05, T=10e3, dimension=1):
        conductivity, Kinf_domain_size = params

        lbda = 2.76
        chi = 1.26e3
        factor = lbda/(1 + lbda)*1/chi

        return run_ML_experiment(
            conductivity=conductivity*factor,
            Kinf_domain_size=Kinf_domain_size,
            N=N,
            dt=dt,
            T=T,
            dimension=dimension,
            verbose=True
        )

    resource_usage = resource.getrusage(resource.RUSAGE_SELF)

    tick = time.time()
    pool = Pool(processes=1)
    identifier_list = pool.map(experiment, parameter_list)
    tock = time.time()

    max_memory_usage = resource_usage.ru_maxrss/1e6  # Kb to Gb
    print("Max memory usage: {:3.1f} Gb".format(max_memory_usage))
    print("Execution time: {:.2f}".format(tock - tick))
