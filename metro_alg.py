from io import TextIOWrapper
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import imageio

epsilon = 1E-014


# Class that holds information and displays it
class Graph:
    # Method to allow people to record data
    def graph_record(self, x, y):
        self.info[0].append(x)
        self.info[1].append(y)

    # Changes the name of the plot
    def rename(self, title=None, x_label=None, y_label=None):
        if title is not None:
            self.title = title
        if x_label is not None:
            self.x_label = x_label
        if y_label is not None:
            self.y_label = y_label

    # Displays the plot to the user
    def graphit(self):
        plt.title(self.title)
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.scatter(self.info[0][:], self.info[1][:])
        plt.show()

    def __init__(self, title="Insert title here", x_label="Iteration", y_label="Energy"):
        self.info = [[], []]
        self.title = title
        self.x_label = x_label
        self.y_label = y_label


class Magnet:
    # Saves things in the graph so that we can remember them for later
    def record_state(self):
        self.energy_graph.graph_record(self.iteration, self.energy)
        self.magnetization_graph.graph_record(self.iteration, 2 * self.num_up - self.x_dim * self.y_dim)
        self.iteration += 1

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

    # Flips the specified spin, and updates the energy
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

    def get_x(self):
        return self.x_dim

    def get_y(self):
        return self.y_dim

    def get_energy(self):
        return self.energy

    def count_ups(self):
        count = 0
        for x_idx, row in enumerate(self.board):
            for y_idx, element in enumerate(row):
                if self.board[x_idx][y_idx] == 1:
                    count += 1
        return count

    def find_energy(self):
        energy = 0
        for x_idx, row in enumerate(self.board):
            for y_idx, element in enumerate(row):
                if x_idx < self.x_dim - 1:
                    energy -= self.board[x_idx, y_idx] * self.board[x_idx + 1, y_idx]
                if y_idx < self.y_dim - 1:
                    energy -= self.board[x_idx, y_idx] * self.board[x_idx, y_idx + 1]
        return energy

    def __init__(self, x, y, preset=[[-2]]):
        # This will record the information we need about energy
        self.energy_graph = Graph(title="Energy vs Iterations")
        # This will record the information we need about magnetization
        self.magnetization_graph = Graph(title="Magnetization vs Iterations")
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
def alg(num_itr, temp, random=True, save_images=False):
    # Create a preset that is up on one half and down on the other
    if not random:
        preset = (-1 * np.ones([100, 100]))
        for i in range(50):
            np.put(preset[i], [np.arange(100)], [1])
        my_mag = Magnet(100, 100, preset)
    # Otherwise generate a random magnet
    else:
        # my_mag = Magnet(1000, 1000)
        my_mag = Magnet(100, 100)

    # Display the magnet and print info
    my_mag.gen_info()
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
    r_r(num_itr, my_mag, temp, save_images, f)
    f.close()

    # Counting up all the times that a boltzmann factor had to be used, to find the average
    if len(output) > 0 and isinstance(output[-1], np.floating):
        boltz_sum += output[-1]
        idx += 1
    # If at least one boltzmann factor was encountered, returns the average
    if idx > 0:
        print("Average of the boltzmann factors is " + str(boltz_sum / idx))

    # Show magnet and print info
    my_mag.gen_info()
    my_mag.display()


if __name__ == '__main__':
    # graph = Graph("Energy by iterations", "Iteration", "Energy")
    # for i in range(10):
    #     graph.record(i, np.exp(i))
    # graph.graphit()

    is_random = True
    save_images = False
    alg(1000, 00.1, is_random, save_images)
    # gif_maker("Images", "order_formation")
