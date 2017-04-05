from forward_model import Reflect


'''
routines:
extract a full satellite track between a set time range
remove the direct signal (5-10 order polynomial)
'''


if __name__ == '__main__':
    R = Reflect('plot_files/mkea0010', 1.8, 0)
    R.plot_omega()
