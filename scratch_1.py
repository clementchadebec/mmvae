import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from src.vis import plot_posteriors

means = [[[2,3],[4,1]],[[1,2],[3,1]]]
stds = [[[2,1],[1,1]],[[1,2],[1,3]]]

#
# fig, axs = plt.subplots(1,1)
# axs.add_patch(Ellipse((0,0), 1,1))
# plt.show()
plot_posteriors(means, stds, 'coucou.png', ticks=[1,2])