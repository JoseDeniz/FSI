import random
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Environment size
width = 5
height = 16

# Actions
num_actions = 4

actions_list = {
    "UP": 0,
    "RIGHT": 1,
    "DOWN": 2,
    "LEFT": 3
}

actions_map = {
    0: "UP",
    1: "RIGHT",
    2: "DOWN",
    3: "LEFT"
}

actions_vectors = {
    "UP": (-1, 0),
    "RIGHT": (0, 1),
    "DOWN": (1, 0),
    "LEFT": (0, -1)
}

# Discount factor 0.8
discount = 0.8

Q = np.zeros((height * width, num_actions))  # Q matrix
Rewards = np.zeros(height * width)  # Reward matrix, it is stored in one dimension


def getState(y, x):
    return y * width + x


def getStateCoord(state):
    return int(state / width), int(state % width)


def getActions(state):
    y, x = getStateCoord(state)
    actions = []
    if x < width - 1:
        actions.append("RIGHT")
    if x > 0:
        actions.append("LEFT")
    if y < height - 1:
        actions.append("DOWN")
    if y > 0:
        actions.append("UP")
    return actions


def getRndAction(state):
    return random.choice(getActions(state))


def getRndState():
    return random.randint(0, height * width - 1)


Rewards[4 * width + 3] = -10000
Rewards[4 * width + 2] = -10000
Rewards[4 * width + 1] = -10000
Rewards[4 * width + 0] = -10000

Rewards[9 * width + 4] = -10000
Rewards[9 * width + 3] = -10000
Rewards[9 * width + 2] = -10000
Rewards[9 * width + 1] = -10000

Rewards[3 * width + 3] = 100
final_state = getState(3, 3)

print np.reshape(Rewards, (height, width))


def qlearning(s1, a, s2):
    Q[s1][a] = Rewards[s2] + discount * max(Q[s2])
    return


movements_average = 0

movements = 0


def greedy(state):
    global movements
    return actions_map[np.argmax(Q[state])] \
        if max(Q[state]) > 0 else getRndAction(state)


def e_greedy(state):
    global e_greedy_movements
    if random.uniform(0.0, 1.0) > 0.9:
        return getRndAction(state)
    return greedy(state)


# Episodes
episodes = 100
list_movements = []
for i in xrange(episodes):
    state = getRndState()
    while state != final_state:
        #action =getRndAction(state)
        action = greedy(state)
        #action = e_greedy(state)
        y = getStateCoord(state)[0] + actions_vectors[action][0]
        x = getStateCoord(state)[1] + actions_vectors[action][1]
        new_state = getState(y, x)
        qlearning(state, actions_list[action], new_state)
        state = new_state
        movements += 1
        movements_average +=1
    print "Number of movements: ", (i, movements)
    list_movements.append(movements)
    movements = 0
print "---------------------------------------"
print "Number of movements average", (movements_average/episodes)
print "---------------------------------------"

plt.ylabel('Movements')
plt.xlabel('Epochs')
movements_line, = plt.plot(list_movements)
plt.legend(handle=[movements_line],
           label=["Movements Line"])
#plt.savefig('greedy.png')
#plt.savefig('e_greedy.png')
plt.savefig('random_movements.png')

# s = 0
# ax = plt.axes()
# ax.axis([-1, width + 1, -1, height + 1])

# for j in xrange(height):
#
#    plt.plot([0, width], [j, j], 'b')
#    for i in xrange(width):
#        plt.plot([i, i], [0, height], 'b')
#
#        direction = np.argmax(Q[s])
#        if s != final_state:
#            if direction == 0:
#                ax.arrow(i + 0.5, 0.75 + j, 0, -0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
#            if direction == 1:
#                ax.arrow(0.25 + i, j + 0.5, 0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
#            if direction == 2:
#                ax.arrow(i + 0.5, 0.25 + j, 0, 0.35, head_width=0.08, head_length=0.08, fc='k', ec='k')
#            if direction == 3:
#                ax.arrow(0.75 + i, j + 0.5, -0.35, 0., head_width=0.08, head_length=0.08, fc='k', ec='k')
#        s += 1
#
#    plt.plot([i + 1, i + 1], [0, height], 'b')
#    plt.plot([0, width], [j + 1, j + 1], 'b')

# plt.show()
