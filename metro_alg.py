from io import TextIOWrapper
import numpy as np
from magnet import Magnet, Graph
from magnet import str2bool
import time

# To avoid divide-by-zero errors
epsilon = 1E-014


def r_r(num_iters, mag, temp, save_images=False, debug_file=None):
    assert debug_file is None or isinstance(debug_file, TextIOWrapper)
    assert isinstance(temp, (np.floating, float, int))
    assert isinstance(num_iters, int)

    counter = 0
    im_idx = 0
    for i in range(num_iters):
        if save_images and counter == 50:
            counter = 0
            r_r_iter(mag, temp, im_idx, debug_file)
            im_idx += 1
        else:
            counter += 1
            r_r_iter(mag, temp, None, debug_file)


# Does one complete iteration of the R&R algorithm
def r_r_iter(mag, temp, im_idx=None, debug=None):
    assert debug is None or isinstance(debug, TextIOWrapper)
    assert im_idx is None or (im_idx, int)
    initial = mag.get_energy()
    x = int(np.random.rand() * mag.get_x())
    y = int(np.random.rand() * mag.get_y())
    mag.flip_spin(x, y)
    if debug is not None:
        debug.write(str(temp) + ", " + str(x) + ", " + str(y) + str(initial - mag.get_energy()))
    fname = "image" + str(im_idx)

    # If the energy is less than 0, go ahead
    if initial > mag.get_energy():
        # Record the new energy
        mag.record_state()
        if debug is not None:
            debug.write("Decreased Energy\n")
    # This is to avoid divide by zero errors
    elif temp < epsilon:
        if debug is not None:
            debug.write("State stayed the same; Temperature was lower than epsilon, ")
        mag.flip_spin(x, y)
        mag.record_state()
    else:
        # Applying Boltzmann statistics
        r = np.random.rand()
        # Using transition probability
        power = ((initial - mag.get_energy()) / temp)
        boltzmann = np.exp(power)
        if r > boltzmann:
            if debug is not None:
                debug.write("State stayed the same; r was larger than the boltzmann factor, ")
            mag.flip_spin(x, y)
            mag.record_state()
        else:
            # Record the new energy
            mag.record_state()
            if debug is not None:
                debug.write("State was changed; r was lower than the boltzmann factor, ")
        if debug is not None:
            debug.write(str(r) + ", " + str(boltzmann) + "\n")
    # Saving the image after all modifications have happened
    if im_idx is not None:
        mag.save_plot(fname)
    return


# Tries to run the rosenbluth algorithm
def alg(num_itr, temp, x, y, stab=500, display=True, random=True, is_saved=False):
    # Create a preset that is up on one half and down on the other
    if not random:
        preset = (-1 * np.ones([100, 100]))
        for i in range(50):
            np.put(preset[i], [np.arange(100)], [1])
        my_mag = Magnet(100, 100, preset)
    # Otherwise generate a random magnet
    else:
        my_mag = Magnet(x, y)

    # Display the magnet and print info
    my_mag.gen_info()
    if display:
        my_mag.display()

    # File to print debug logs into. Will overwrite past error logs
    f = open("log.csv", "w")
    output = []
    # Write a first line (useful to remember what each column is)
    f.write(str(["temp", "x", "y", "delta_E", "What Happened", "random value", "Boltzmann factor"]) + '\n')
    # Variables to find the average of the boltzmann sums
    boltz_sum = 0
    idx = 0

    # Runs the algorithm for a given number of iterations, magnet, and temperature
    r_r(num_itr, my_mag, temp, is_saved, f)
    f.close()

    # Counting up all the times that a boltzmann factor had to be used, to find the average
    if len(output) > 0 and isinstance(output[-1], np.floating):
        boltz_sum += output[-1]
        idx += 1
    # If at least one boltzmann factor was encountered, returns the average
    if idx > 0:
        print("Average of the boltzmann factors is " + str(boltz_sum / idx))

    # Show magnet and print info
    # print(my_mag.energy_graph.find_exp_val(ignore=2000))
    # print(my_mag.energy_graph.find_var(ignore=2000))
    my_mag.gen_info()
    print("Heat Capacity of the magnet is: " + str(my_mag.heat_cap(stab)))
    print("Magnetic Susceptibility of the magnet is: " + str(my_mag.susceptibility(stab)))
    print("\n")
    print(my_mag.heat_cap(stab))
    print(my_mag.susceptibility(stab))
    if display:
        my_mag.display()
        my_mag.display_state()


# This method calculates thermodynamics quantities for a range of temperatures. As such, it can take quite a while!
def var_temp(x, y, iter=1000, stab=500, temperatures=None, repl=False):
    # This variable is a list of magnets. After obtaining information at every temperature, each magnet will be stored
    # in here, and at the end there will be a REPL to debug what happened at every temperature.
    debug = []
    # If the user hasn't specified a temperature range, we'll use this preset one from T=0.4 to 4 with step size 0.1
    if temperatures is None:
        temperatures = np.arange(0.4, 4.1, 0.01)
        # temperatures = np.arange(0.4, 4.1, 3)

    # Total number of iterations
    tot_its = len(temperatures)
    # These graphs will keep track of heat capacity, magnetic susceptibility, average energy, and average magnetization
    heat_graph = Graph(title="Heat Capacity as a Function of Temperature", x_label="Temperature", y_label="Heat Capacit\
y")
    sus_graph = Graph(title="Magnetic Susceptibility as a Function of Temperature", x_label="Temperature", y_label="Mag\
netic susceptibility")
    nrg_graph = Graph(title="Energy as a Function of Temperature", x_label="Temperature", y_label="Energy")
    mag_graph = Graph(title="Magnetization as a Function of Temperature", x_label="Temperature", y_label="Magnetization")

    # Begins iterating through each temperature
    for i, temp in enumerate(temperatures):
        # Useful to know how far through we are
        print("iteration " + str(i) + " out of " + str(tot_its))
        # Creates a magnet that will be used in this temperature iteration
        mag = Magnet(x, y)
        # iterates over the algorithm iter times
        for idx in range(iter):
            r_r_iter(mag, temp)

        # if the user wants to debug at the end of the algorithm, they can set repl to True which will allow them
        # to step through magnet of different temperatures
        if repl:
            debug.append(mag)

        # Once it's done iteration, thermodynamic quantities of the magnet are recorded
        nrg_graph.graph_record(temp, mag.get_avg_energy(stab=stab))
        mag_graph.graph_record(temp, mag.get_avg_magnetization(stab=stab))
        heat_graph.graph_record(temp, mag.heat_cap(temp, stab=stab))
        sus_graph.graph_record(temp, mag.susceptibility(temp, stab=stab))

    # This is a useful caption to keep track of images
    label = str(x) + " by " + str(y) + " magnet, run through " + str(iter) + " iterations, with stability estimated to \
occur at " + str(stab) + " iterations"
    # Shows useful graphs of the energy, magnetization, heat capacity, and magnetic susceptibility at the end of the
    # cycles
    nrg_graph.graphit(caption=label)
    mag_graph.graphit(caption=label)
    heat_graph.graphit(caption=label)
    sus_graph.graphit(caption=label)

    # if the function explicitly specified that a REPL was not going to happen, there is no point in going further in
    # this function.
    if not repl:
        return

    # this is the debugging REPL. It should (if I'm not lazy and manage to implement it all the way through) let you
    # examine magnets held at different temperatures after they went through the R&R algorithm
    print("Entering the REPL. Lets you examine magnets of different temperatures. Type \"Quit\" or \"Exit\" if you wish\
 to, well, exit")
    while True:
        # total number of magnets
        num_mags = len(debug)
        buf = input("There are " + str(num_mags) + " Magnets. Enter a value between 0 and " + str(num_mags) + " to \
see information about that magnet")
        low = buf.lower()
        # Quits the REPL if the user specifies
        if low == "quit" or low == "exit":
            break
        # Error checking the input
        elif isinstance(buf, int):
            print("Please enter a valid integer")
            time.sleep(0.3)
            continue
        elif int(buf) > num_mags:
            print("integer is too large")
            time.sleep(0.3)
            continue

        # the magnet of interest
        cool_mag = debug[int(buf)]
        cool_mag.gen_info()
        cool_mag.display()


if __name__ == '__main__':
    # Change in energy is a multiple of 4 (0, 4, 8)
    # Change in magnetization is a multiple of 2 (-2 or 2)
    # More than 1000000 (1 million) iterations for var_temp takes too long
    # temps = np.ones(30) * 2
    var_temp(10, 10, iter=100000, stab=2000, repl=True)
    # is_random = True
    # save_images = False
    # alg(10000, 0.4, 10, 10, display=False, stab=2000)
