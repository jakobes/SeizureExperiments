import resource
import warnings
import time

from multiprocessing import Pool
from itertools import product
from pathlib import Path

from single_experiment import (
    run_ML_experiment,
    run_ode_step_experiment,
)

from make_report import make_report

warnings.simplefilter("ignore", UserWarning)


if __name__ == "__main__":
    conductivity_list = (1/128, 1/256)
    KL_list = (0.5, 0.5 / 2, 0.5 / 4)
    parameter_list = product(conductivity_list, KL_list)

    def experiment(params, N=300, dt=0.05, T=100e3, dimension=1, ode_step_fraction=1):
        conductivity, Kinf_domain_size = params
        return run_ML_experiment(
            conductivity,
            Kinf_domain_size,
            N,
            dt,
            T,
            dimension,
            ode_step_fraction,
            reload=False
        )

    resource_usage = resource.getrusage(resource.RUSAGE_SELF)

    tick = time.time()
    pool = Pool(processes=6)
    identifier_list = pool.map(experiment, parameter_list)
    tock = time.time()

    max_memory_usage = resource_usage.ru_maxrss/1e6  # Kb to Gb
    print("Max memory usage: {:3.1f} Gb".format(max_memory_usage))
    print("Execution time: {:.2f}".format(tock - tick))

    for iden in identifier_list:
        casedir = Path("out_cressman") / iden
        make_report(casedir, dim=1)
