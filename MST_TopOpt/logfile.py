import numpy as np

def init_dir(str):

    import os
    import datetime

    now = datetime.datetime.now()
    today = now.strftime("%Y-%m-%d_%H-%M-%S"+str)
    directory = f"{today}"
    if not os.path.exists(directory):
        os.makedirs(directory)

    return directory, today

def write_line(f):
    f.write("--------------------------------------------- \n")

def write_general_info(f, sim):
    write_line(f)
    f.write("Simulation domain data: \n")
    write_line(f)
    f.write("Scaling: {:.6e}  \n".format(sim.scaling))
    f.write("Number of elements in Y: {:.6e}  \n".format(sim.nElY))
    f.write("Number of elements in X: {:.6e}  \n".format(sim.nElX))
    write_line(f)
    f.write("Material parameters: \n")
    write_line(f)
    f.write("Permittivity: {:.6e}  \n".format(sim.eps))
    f.write("Permittivity particle: {:.6e}  \n".format(sim.eps_part))
    write_line(f)

def create_logfile_optimization(sim,idx_RHS_EM = None, val_RHS_EM=None, idx_RHS_heat = None, val_RHS_heat=None):
    import numpy as np
    directory = sim.directory_opt
    today = sim.today

    logfile = f"{directory}/opt_logfile_{today}.txt"
    with open(logfile, "w") as f:
        f.write(f"Optimization log data for {today}:\n")
        write_general_info(f, sim)
        f.write("Topology Optimization parameters: \n")
        write_line(f)
        f.write("Threshold level, eta: {:.6e}  \n".format(sim.eta))
        f.write("Threshold strength, beta: {:.6e}  \n".format(sim.beta))
        f.write("Filter radius: {:.6e}  \n".format(sim.fR))
        f.write("Maximum number of iterations: {:.6e}  \n".format(sim.maxItr))
        f.write(f"Continuation scheme: {sim.continuation_scheme}  \n")
        np.save(f"{directory}/dVini.npy", sim.dVini)
        np.save(f"{directory}/dVini_part.npy", sim.dVini_part)
        np.save(f"{directory}/dVs.npy", sim.dVs)
        np.save(f"{directory}/dVs_part.npy", sim.dVs_part)
        write_line(f)

        sim.iteration_history(it_num = sim.maxItr, save=True, dir=directory)