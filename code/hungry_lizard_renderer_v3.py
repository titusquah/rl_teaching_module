import numpy as np
from hungry_lizard_v2 import HungryLizard_v2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors


def hungry_lizard_renderer_v3(action_list):
    lizard = HungryLizard_v2()
    current_reward = 0
    height = lizard.height
    width = lizard.width

    def flatten(mini_x, mini_y):
        return int(mini_x + mini_y * width)

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
    
    def x_reflect(mini_x):
        if mini_x > (width - 1) / 2:
            new_x = int((width - 1) / 2
                        - (mini_x - (width - 1) / 2))
        elif mini_x < (width - 1) / 2:
            new_x = int(
                int(np.ceil((width - 1) / 2))
                + int((width - 1) / 2 - mini_x))
        else:
            new_x = mini_x
        return new_x

    map_state_list = []
    reward_list = []
    bird_loc_list = []
    small_loc_list = []
    large_loc_list = []
    lizard_loc_list = []
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
        x, y = stack(lizard.state[0])
        y = y_reflect(y)
        map_state[y, x] = 4
        lizard_locs.append([x, y])
        map_state_list.append(map_state)
        bird_loc_list.append(bird_locs)
        small_loc_list.append(small_locs)
        large_loc_list.append(large_locs)
        lizard_loc_list.append(lizard_locs)
        reward_list.append(current_reward)
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = colors.ListedColormap(['white',
                                  '#e41a1c',
                                  '#984ea3',
                                  '#377eb8',
                                  '#4daf4a'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    ax.grid(which='major', axis='both', linestyle='-', color='k',
            linewidth=2)
    ax.tick_params(
        axis='both',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=False,
        labelleft=False,
        left=False,
        right=False)  # labels along the bottom edge are off
    x_ticks = np.arange(-0.5, width - 0.5, 1)
    y_ticks = np.arange(-0.5, height - 0.5, 1)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.set_xlim(-0.5, width - 0.5)
    ax.set_ylim(-0.5, height - 0.5)
    im = ax.imshow(np.zeros((width, height)),
                   animated=True,
                   cmap=cmap, norm=norm)
    text_locs = []

    for ind2 in range(width):
        for ind3 in range(height):
            text_locs.append([ind2, ind3])
    texts1 = [ax.text(loc[0], loc[1], '',
                      ha='center',
                      va='center',
                      fontsize=18) for loc in text_locs]
    texts2 = [ax.text(1.02, 0,
                      'L = Lizard\nB = Bird\n3 = 3 crickets\n10 = 10 crickets',
                      transform=ax.transAxes,
                      fontsize=18),
              ax.text(1.02, 1,
                      'Total reward = {0:.2f}'.format(0),
                      transform=ax.transAxes,
                      fontsize=18,
                      va='top'
                      )]
    for ind2 in range(20):
        plt.tight_layout()

    def animate(ind4):
        new_map_state = np.zeros((width, height))
        for ind5 in range(height):
            new_y = y_reflect(ind5)
            new_y = ind5
            for ind6 in range(width):
                new_x = x_reflect(ind6)
                new_map_state[new_x, new_y] = map_state_list[ind4][ind6, ind5]
        im.set_array(new_map_state)
        bird_locs = bird_loc_list[ind4]
        reward = reward_list[ind4]
        small_locs = small_loc_list[ind4]
        large_locs = large_loc_list[ind4]
        lizard_locs = lizard_loc_list[ind4]
        texts2[1].set_text('Total reward = {0:.0f}'.format(reward))
        colored_tiles = []
        for bird_loc in bird_locs:
            new_y = y_reflect(bird_loc[1])
            ind5 = flatten(new_y, bird_loc[0])
            texts1[ind5].set_text('B')
            colored_tiles.append(ind5)
        for small_loc in small_locs:
            new_y = y_reflect(small_loc[1])
            ind5 = flatten(new_y,small_loc[0])
            texts1[ind5].set_text('3')
            colored_tiles.append(ind5)
        for large_loc in large_locs:
            new_y = y_reflect(large_loc[1])
            ind5 = flatten(new_y, large_loc[0])
            texts1[ind5].set_text('10')
            colored_tiles.append(ind5)
        for lizard_loc in lizard_locs:
            new_y = y_reflect(lizard_loc[1])
            ind5 = flatten(new_y, lizard_loc[0])
            texts1[ind5].set_text('L')
            colored_tiles.append(ind5)
        for ind5 in range(width * height - 1):
            if ind5 not in colored_tiles:
                texts1[ind5].set_text('')
        return im,

    ani = animation.FuncAnimation(fig, animate, frames=len(action_list) + 1,
                                  interval=500, blit=False)
    return ani
