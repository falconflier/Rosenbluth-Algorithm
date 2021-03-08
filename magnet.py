import argparse
from io import TextIOWrapper
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import statistics as stats
import glob
import imageio


# Class that holds graph information and displays it. Can also calculated standard deviation and mean.
class Graph:
    # Finds the standard deviation of the y-components of its data, ignoring the first few values (as specified by the
    # "stab" variable)
    def get_std_dev(self, stab=0):
        # Error checking
        if stab > len(self.info[1]):
            stab = 0
            print("Stabilization estimate exceeded max number of iterations. Setting to 0")
        return stats.stdev(self.info[1][stab:])

    # Finds the mean of the y-component of its data, ignoring the first few values as specified by the stab argument
    def get_mean(self, stab=0):
        # Error Checking
        if stab > len(self.info[1]):
            stab = 0
            print("Stabilization estimate exceeded max number of iterations. Setting to 0")
        return stats.mean(self.info[1][stab:])

    # Method to allow people to record data in the graph
    def graph_record(self, x, y):
        self.info[0].append(x)
        self.info[1].append(y)

    # Changes the name and axis labels of the plot
    def rename(self, title=None, x_label=None, y_label=None):
        if title is not None:
            self.title = title
        if x_label is not None:
            self.x_label = x_label
        if y_label is not None:
            self.y_label = y_label

    # Displays the plot to the user. Caption is a string which can display useful information below the graph. I found
    # captions useful when I wanted to display the temperature, number of iterations, or size of the magnet that the
    # graph was associated with
    def graphit(self, caption=None):
        if caption is not None:
            # plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=12)
            plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=5)

        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.scatter(self.info[0][:], self.info[1][:])
        plt.show()

    # Initializing a Graph. All of the data is held in the list of lists self.info
    def __init__(self, title="Insert title here", x_label="Iteration", y_label="Energy"):
        self.info = [[], []]
        self.title = title
        self.x_label = x_label
        self.y_label = y_label


# This is where all of the information about the magnet is stored. This class also has lots of methods that calculate
# relevant thermodynamic quantities, and methods to store data for later viewing.
class Magnet:
    # The functions at the beginning of the class mostly deal with calculating energy, magnetization, heat capacity,
    # susceptibility, and the average of those quantities. Averages are calculated by using an associated instance
    # of the Graph class, which stores past information which can be averaged over at a later point.

    # Saves the energy and magnetization at a given iteration in the graph so that we can remember them for later
    def record_state(self, stab=500):
        self.energy_graph.graph_record(self.iteration, self.energy)
        self.magnetization_graph.graph_record(self.iteration, self.magnetization())
        self.iteration += 1

    # Finds the magnetization of the magnet (num up - num down = 2 * num up - total spins)
    def magnetization(self):
        return 2 * self.num_up - self.x_dim * self.y_dim

    # Returns the average value of the y-component of the magnetization data, ignoring the first few data points
    def get_avg_magnetization(self, stab=500):
        return self.magnetization_graph.get_mean(stab)

    # This method returns the field stored in self.energy. It should return the correct energy, and in a much shorter
    # time than find_energy(), but depends on everything else correctly updating self.energy.
    def get_energy(self):
        return self.energy

    # This method finds the average energy by using the data stored in self.energy_graph, ignoring the first few data
    # points
    def get_avg_energy(self, stab=500):
        # print("average energy is: " + str(self.energy_graph.get_mean(stab)))
        return self.energy_graph.get_mean(stab)

    # This method iterates through the entire magnet and calculates the energy of the whole system. It is used whenever
    # a new magnet is created, and should never return a bad value
    def find_energy(self):
        energy = 0
        for x_idx, row in enumerate(self.board):
            for y_idx, element in enumerate(row):
                if x_idx < self.x_dim - 1:
                    energy -= self.board[x_idx, y_idx] * self.board[x_idx + 1, y_idx]
                if y_idx < self.y_dim - 1:
                    energy -= self.board[x_idx, y_idx] * self.board[x_idx, y_idx + 1]
        return energy

    # This method returns the heat capacity of the system, ignoring the first few data points
    def heat_cap(self, temp, stab=500):
        return self.energy_graph.get_std_dev(stab) / (temp ** 2)

    # This method returns the magnetic susceptibility of the system, ignoring the first few data points
    def susceptibility(self, temp, stab=500):
        return self.magnetization_graph.get_std_dev(stab) / temp

    # Shows the magnetization and energy graphs
    def display_state(self):
        self.energy_graph.graphit()
        self.magnetization_graph.graphit()

    # Shows the magnet as a matplotlib display, with green as up and blue as down
    def display(self):
        # Blue is down, green is up
        cmap = colors.ListedColormap(['blue', 'green'])
        bounds = [-1, 0, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(self.board, cmap=cmap, norm=norm)

        plt.show()

    # This method saves the graphical representation of the magnet to the file given in fname
    def save_plot(self, fname):
        # Blue is down, green is up
        cmap = colors.ListedColormap(['blue', 'green'])
        bounds = [-1, 0, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        fig, ax = plt.subplots()
        ax.imshow(self.board, cmap=cmap, norm=norm)
        fname = "./Images/" + fname
        plt.savefig(fname)
        plt.close()

    # Gives us general information about the current state of the magnet
    def gen_info(self):
        print("%%%%%%%%%%%%%%%%%%%%% General Information About Magnet: %%%%%%%%%%%%%%%%%%%%%")
        num_spins = self.x_dim * self.y_dim
        print("Board is of size " + str(num_spins) + " with x-dimension " + str(self.x_dim) + " and \
y-dimension " + str(self.y_dim))
        print(str(self.num_up) + " spins are up (" + str(self.num_up / num_spins * 100) + "% of the total)")
        print("Total energy of the magnet is " + str(self.energy) + '\n')

    # Returns the size of the magnet in the x-dimension
    def get_x(self):
        return self.x_dim

    # Returns the size of the magnet in the y-dimension
    def get_y(self):
        return self.y_dim

    # Flips the specified spin, and updates the energy. This does NOT implement the metropolis algorithm. When the
    # Metropolis algorithm tests the energy of a certain spin configuration, it calls this method, checks the updated
    # energy, and either leaves it like that or flips it back to the original state.
    def flip_spin(self, x, y):
        if x < 0 or x >= self.x_dim or y < 0 or y >= self.y_dim:
            print("invalid index")
            return
        # Flips the spin with specified index and recounts the number of up spins
        self.board[x, y] *= -1
        self.num_up += self.board[x, y]
        # Recalculating the energy of the magnet
        if x < self.x_dim - 1:
            self.energy -= self.board[x, y] * self.board[x + 1, y] * 2
        if y < self.y_dim - 1:
            self.energy -= self.board[x, y] * self.board[x, y + 1] * 2
        if x > 0:
            self.energy -= self.board[x, y] * self.board[x - 1, y] * 2
        if y > 0:
            self.energy -= self.board[x, y] * self.board[x, y - 1] * 2

    # Counts the total number of up spins by iterating through the entire board. Like find_energy(), it's fairly fool-
    # proof, but slower than just relying on self.num_ups
    def count_ups(self):
        count = 0
        for x_idx, row in enumerate(self.board):
            for y_idx, element in enumerate(row):
                if self.board[x_idx][y_idx] == 1:
                    count += 1
        return count

    # Generates a random mangnet of x by y dimensions, or uses a preset (I mostly ignore this feature though, and it's
    # kind of since I stopped updating it a while ago, so I don't recommend playing around with it).
    def __init__(self, x, y, preset=[[-2]]):
        # This will record the information we need about energy
        self.energy_graph = Graph(title="Energy vs Iterations", x_label="Iteration", y_label="Energy")
        # This will record the information we need about magnetization
        self.magnetization_graph = Graph(title="Magnetization vs Iterations", x_label="Iteration", y_label="Magnetizati\
on")
        # This will give us the x-coordinate of the graph
        self.iteration = 0

        if type(x) != int or type(y) != int or x < 0 or y < 0:
            print("invalid starting dimensions")
            return
        self.x_dim = x
        self.y_dim = y
        if preset[0][0] == -2:
            print("Generating random values")
            self.board = np.ones([x, y])
            for row_index, row in enumerate(self.board):
                for idx, element in enumerate(row):
                    if np.random.rand() < 0.5:
                        row[idx] = -1
        else:
            if len(preset) != x or len(preset[0]) != y:
                print("preset value does not match indicated dimensions. Please try again")
            print("Using preset values")
            self.board = preset
        self.num_up = self.count_ups()
        self.energy = self.find_energy()


# Taken from the web. See if something is an integer
def is_int(val):
    try:
        num = int(val)
    except ValueError:
        return False
    return True


# This is a useful method to see if a given string is trying to indicate a True or False Boolean. Taken from
# stackexchange
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# Method that was supposed to take in a bunch of images and output a gif. It's taken pretty much directly from the
# internet, but I never managed to get it to produce fun videos, which is a bit of a shame.
def gif_maker(dir, name):
    print(dir + "/*.png")
    file_list = glob.glob(dir + "/*.png")
    print(file_list)
    # From this post: https://stackoverflow.com/questions/753190/programmatically-generate-video-or-animated-gif-in
    # -python

    # images = []
    # for filename in file_list:
    #     # print(filename)
    #     images.append(imageio.imread(filename))
    # ? imageio.mimsave()
    # # imageio.mimsave('./' + name + '.gif', format='GIF', fps=30, ims=1)

    name = './' + name + '.gif'
    print(name)
    with imageio.get_writer(name, mode='I') as writer:
        for filename in file_list:
            # print(filename)
            image = imageio.imread(filename)
            writer.append_data(image)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Everything below this line is old methods that aren't used anymore, but which are good references for me so I don't
# want to delete them.

# Tries to run the rosenbluth algorithm. This is an old method that I didn't want cluttering up the algorithm file.
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


# I can't do a circular import, so I created this dummy method to avoid problems in r_r()
def r_r_iter(mag, temp, im_idx, debug_file):
    pass
