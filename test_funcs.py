

def test_save():
    mag = Magnet(10, 10)
    mag.display()
    mag.save_plot("test")


def test_func():
    my_mag = Magnet(10, 10)
    my_mag.gen_info()
    my_mag.display()
    my_mag.flip_spin(0, 0)
    my_mag.display()
    my_mag.gen_info()
    my_mag.flip_spin(1, 0)
    my_mag.display()
    my_mag.gen_info()


def basic():
    my_mag = Magnet(2, 2)
    my_mag.gen_info()
    my_mag.display()
    my_mag.flip_spin(0, 0)
    my_mag.gen_info()
    print(my_mag.find_energy())
    my_mag.display()


def pre_built():
    my_mag = Magnet(10, 10, np.zeros([10, 10]))
    my_mag.display()