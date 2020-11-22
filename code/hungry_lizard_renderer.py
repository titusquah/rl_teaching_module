import numpy as np
from hungry_lizard import HungryLizard
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors


def hungry_lizard_renderer(action_list):
    lizard = HungryLizard()
    current_reward = 0
    height = lizard.height
    width = lizard.width

    def stack(ind):
        mini_x = int(ind % width)
        mini_y = int(ind // width)
        return mini_x, mini_y

    def y_reflect(mini_y):
        if mini_y > (height - 1) / 2:
            new_y = int((height - 1) / 2
                        - (mini_y - (height - 1) / 2))
        elif mini_y < (height - 1) / 2:
            new_y = int(
                int(np.ceil((height - 1) / 2))
                + int((height - 1) / 2 - mini_y))
        else:
            new_y = mini_y
        return new_y

    ims = []
    fig, ax = plt.subplots(figsize=(8, 5))
    for ind1 in range(len(action_list) + 1):
        if ind1 != 0:
            state, reward, done, info = lizard.step(action_list[ind1 - 1])
            current_reward += reward
        bird_locs = []
        small_locs = []
        large_locs = []
        lizard_locs = []

        map_state = np.zeros((width, height))
        for bird_loc in lizard.birds_loc:
            x, y = stack(bird_loc)
            y = y_reflect(y)

            map_state[y, x] = 1
            bird_locs.append([x, y])

        for small_loc in lizard.small_reward_loc:
            x, y = stack(small_loc)
            y = y_reflect(y)
            map_state[y, x] = 2
            small_locs.append([x, y])
        for large_loc in lizard.large_reward_loc:
            x, y = stack(large_loc)
            y = y_reflect(y)
            map_state[y, x] = 3
            large_locs.append([x, y])
        x, y = stack(lizard.state)
        y = y_reflect(y)
        map_state[y, x] = 4
        lizard_locs.append([x, y])

        # create discrete colormap
        cmap = colors.ListedColormap(['white',
                                      '#e41a1c',
                                      '#984ea3',
                                      '#377eb8',
                                      '#4daf4a'])
        bounds = [0, 1, 2, 3, 4, 5]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        plt.cla()
        ax.imshow(map_state, cmap=cmap, norm=norm)
        locs = [bird_locs, small_locs, large_locs, lizard_locs]
        letters = ['B', '3', '10', 'L']
        for i in range(len(locs)):
            mini_locs = locs[i]
            letter = letters[i]
            for loc in mini_locs:
                ax.annotate(letter, loc,
                            ha='center',
                            va='center',
                            fontsize=18)
        ax.annotate('L = Lizard\nB = Bird\n3 = 3 crickets\n10 = 10 crickets',
                    (1.02, 0),
                    xycoords='axes fraction',
                    fontsize=18)
        ax.annotate('Current reward = {0}'.format(current_reward),
                    (1.02, 1),
                    xycoords='axes fraction',
                    fontsize=18,
                    va='top')
        # draw gridlines
        ax.grid(which='major', axis='both', linestyle='-', color='k',
                linewidth=2)
        x_ticks = np.arange(-0.5, width - 0.5, 1)
        y_ticks = np.arange(-0.5, height - 0.5, 1)
        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        for i in range(5):
            plt.tight_layout()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        im = plt.imshow(data, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims, interval=500, blit=True,
                              repeat_delay=1000)
    return ani

