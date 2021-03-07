from io import TextIOWrapper
import numpy as np
from magnet import Magnet

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
            r_r_help(mag, temp, im_idx, debug_file)
            im_idx += 1
        else:
            counter += 1
            r_r_help(mag, temp, None, debug_file)


def r_r_help(mag, temp, im_idx=None, debug=None):
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
def alg(num_itr, temp, x, y, random=True, is_saved=False):
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
    # my_mag.gen_info()
    # my_mag.display()

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
    # my_mag.gen_info()
    print(my_mag.heat_cap(2000))
    print(my_mag.susceptibility(2000))
    my_mag.display()
    my_mag.display_state()


if __name__ == '__main__':
    is_random = True
    save_images = False
    alg(100000, 0.01, 100, 100, is_random, save_images)
