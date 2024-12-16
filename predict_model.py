import tensorflow as tf
import numpy as np
import random
import resources as jq
import pandas as pd
from collections import deque
import pickle
from time import time
import matplotlib.pyplot as plt

# from pyautogui import press, sleep
# while True:
#     sleep(20)
#     press(['left','right','up','down'][random.randrange(0,4)])


class JewelQuest3Env:
    def __init__(self, grid, active_gems):
        self.board, self.golden = jq.generate_state(grid, active_gems, False)

        self.done = False
        self.moves = []
        self.total_matches = 0
        m, n = grid.shape
        for pos in np.ndindex(grid.shape):
            if pos[0] + 1 < m:
                self.moves.append((pos, (pos[0] + 1, pos[1])))
            if pos[1] + 1 < n:
                self.moves.append((pos, (pos[0], pos[1] + 1)))

    def reset(self):
        self.board, self.golden = jq.generate_state(grid, active_gems, False)
        self.done = False
        self.total_matches = 0
        return np.stack((self.board, self.golden), axis=-1)

    def predict_mcts(self):
        ret = jq.MCTreeNode(self.board, self.golden).step(15)
        return env.moves.index(ret.next_move())

    def step(self, move_index):
        swap1, swap2 = self.moves[move_index]
        self.total_matches += 1
        if self.done or self.board[swap1] < 0 or self.board[swap2] < 0:
            # If the move is invalid, punish the model
            return np.stack((self.board, self.golden), axis=-1), -100, self.done
        active_gems = np.unique(self.board)
        active_gems = active_gems[active_gems > 0]
        distribution = jq.calculate_distribution(active_gems)
        self.board[swap1], self.board[swap2] = self.board[swap2], self.board[swap1]
        # Continue matching gem positions while they exist (cascades)
        matched = {'First run'}
        swapped_once = False
        reward = 0
        while len(matched) != 0:
            matched = set()
            for pos, gem in np.ndenumerate(self.board):
                if gem <= 0:
                    continue

                for adj in ((0, 1), (1, 0)):
                    adj1, adj2 = (pos[0] + adj[0], pos[1] + adj[1]), (pos[0] + adj[0] * 2, pos[1] + adj[1] * 2)
                    in_area = 0 <= adj2[0] < m and 0 <= adj2[1] < n
                    if in_area and self.board[adj1] == gem and self.board[adj2] == gem:
                        for matchPos in (pos, adj1, adj2):
                            matched.add(matchPos)
                            swapped_once = True

            # Sort matched positions by height so gravity doesn't affect anything
            matched = sorted(matched, reverse=True)
            while matched:
                pos = matched.pop()
                if self.golden[pos] == 0:
                    self.golden[pos] = 1
                    reward += 1
                # Shift down all elements above the match positions and add random gems to the top
                while 0 <= pos[0] - 1 < m and self.board[pos[0] - 1, pos[1]] > 0:
                    self.board[pos] = self.board[pos[0] - 1, pos[1]]
                    self.board[pos[0] - 1, pos[1]] = 0
                    pos = (pos[0] - 1, pos[1])
                self.board[pos] = random.choices(active_gems, distribution)[0]
        if not swapped_once:
            # Punish model for making an invalid match
            return np.stack((self.board, self.golden), axis=-1), -100, self.done

        # self.done = (self.golden == 1).sum() == (self.board > -1).sum()
        # if self.done:
        #     # Reward model for completion, more reward if less moves made
        #     return np.stack((self.board, self.golden), axis=-1), max(10000 - self.total_matches * 5, 0), self.done
        # Always punish model by 1 every move so that it tries to solve in the least amount
        # return np.stack((self.board, self.golden), axis=-1), reward - 1, self.done

        # Just reward model for making a successful move and quit for now
        return np.stack((self.board, self.golden), axis=-1), reward, True


class DQNAgent:
    """
    Code copy and pasted from chatgpt. It works well.
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = create_dqn_model(input_size, output_size)
        self.target_model = create_dqn_model(input_size, output_size)
        self.update_target_model()
        self.replay_buffer = deque(maxlen=700)
        # Lower gamma is more greedy than a higher gamma (Lower = immediate reward, Higher = equal weight)
        self.gamma = 0.995
        # Epsilon is the exploration probability. Decay is how much it decreases.
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.998

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        # Randomly pick between exploration and exploitation (epsilon threshold)
        if random.random() <= self.epsilon:
            # "Explore" a random output
            return random.randint(0, self.output_size - 1)
        # Predict Q-values for the given state, exploit the highest value
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        return np.argmax(q_values[0])

    def predict(self, state, show_qvals=False):
        q_values = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
        action = np.argmax(q_values[0])

        if show_qvals:
            print("Predicted Q-values:")
            for move, q_val in zip(env.moves, q_values[0]):
                print(move, q_val)
            print("Chosen Action:", env.moves[action])
        return env.moves[action]

    def replay(self, batch_size):
        # Don't train if there aren't enough samples for a batch
        if len(self.replay_buffer) < batch_size:
            return
        # Samples a random batch of experiences
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0), verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0), verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t[0])
            # Train model on new Q-values
            self.model.fit(np.expand_dims(state, axis=0), target, epochs=1, verbose=0)

        # Decay the exploration probability (epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def create_dqn_model(input_shape, output_shape):
    input_layer = tf.keras.layers.Input(input_shape)
    normal_layer = tf.keras.layers.Normalization()(input_layer)

    horiz_layer = tf.keras.layers.Conv2D(16, (2, 3), activation='relu')(normal_layer)
    horiz_flat = tf.keras.layers.Flatten()(horiz_layer)
    verti_layer = tf.keras.layers.Conv2D(16, (3, 2), activation='relu')(normal_layer)
    verti_flat = tf.keras.layers.Flatten()(verti_layer)
    match_layer = tf.keras.layers.Concatenate()([horiz_flat, verti_flat])

    dense_layer = tf.keras.layers.Dense(256, activation='relu')(match_layer)
    dropout_layer = tf.keras.layers.Dropout(0.1)(dense_layer)
    output_layer = tf.keras.layers.Dense(output_shape, activation='linear')(dropout_layer)

    # Input, Normal, Conv2D, Conv2D, Flatten, Flatten, Concat, Dense, Dropout, Dense
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


# grid = np.asarray([
#     [-1,  0,  0,  0,  0,  0,  0, -1],
#     [ 0,  0,  0,  0,  0,  0,  0,  0],
#     [ 0,  0,  0,  0,  0,  0,  0,  0],
#     [ 0,  0,  0,  0,  0,  0,  0,  0],
#     [ 0,  0,  0,  0,  0,  0,  0,  0],
#     [ 0,  0,  0,  0,  0,  0,  0,  0],
#     [ 0,  0,  0,  0,  0,  0,  0,  0],
#     [-1,  0,  0,  0,  0,  0,  0, -1]
# ])
# active_gems = [2, 3, 4, 5]
grid = np.asarray([
    [-1,  0,  0,  0, -1],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0],
    [-1,  0,  0,  0, -1],
])
active_gems = [2, 3, 4]
env = JewelQuest3Env(grid, active_gems)

m, n = grid.shape
agent = DQNAgent(input_size=(m, n, 2), output_size=len(env.moves))
try:
    agent.model = tf.keras.models.load_model('uxmal_model.keras')
except ValueError:
    pass

try:
    agent.replay_buffer = pickle.load(open('uxmal_replay.pkl', 'rb'))
except FileNotFoundError:
    pass

adapt_data = np.stack([np.stack(jq.generate_state(grid, active_gems), axis=-1) for _ in range(2000)])
normalization_layer = agent.model.layers[1]
normalization_layer.adapt(adapt_data)

swap_data = np.zeros((4, *grid.shape))
# 0=model_right, 1=model_down, 2=mcts_right, 3=mcts_down

for i in range(1000):
    if i % 100 == 0:
        print(f"Simulation Finished: {i}")
    state = JewelQuest3Env(grid, active_gems)
    predict_board, predict_golden = state.board, state.golden
    predict_state = np.stack((predict_board, predict_golden), axis=-1)
    # print(predict_board)
    # print(predict_golden)
    swap1, swap2 = agent.predict(predict_state)
    if swap2[1] > swap1[1]:  # swap right
        swap_data[0][swap1] += 1
    else:  # swap down
        swap_data[1][swap1] += 1
    swap1, swap2 = state.moves[state.predict_mcts()]
    if swap2[1] > swap1[1]:  # swap right
        swap_data[2][swap1] += 1
    else:  # swap down
        swap_data[3][swap1] += 1
    # print(swap1, swap2)
    # if type(jq.mcts_simulate_move(predict_board, predict_golden, active_gems, swap1, swap2)[0]) is int:
    #     print('invalid move.')


def plot_text(arr):
    m, n = arr.shape
    for i in range(m):
        for j in range(n):
            plt.text(j, i, int(arr[i, j]), ha="center", va="center", color="tab:red")


plt.subplot(2, 2, 1)
plt.imshow(swap_data[0])
plot_text(swap_data[0])
plt.title('Swap Right')
plt.ylabel('Model')
plt.subplot(2, 2, 2)
plt.imshow(swap_data[1])
plot_text(swap_data[1])
plt.title('Swap Down')
plt.subplot(2, 2, 3)
plt.imshow(swap_data[2])
plot_text(swap_data[2])
plt.ylabel('Monte-Carlo')
plt.subplot(2, 2, 4)
plt.imshow(swap_data[3])
plot_text(swap_data[3])
plt.show()
quit()

episodes = 10000
batch_size = 64
df = pd.DataFrame({'Episode': [], 'Reward': [], 'Epsilon': [], 'Episode Length': [], 'Time Elapsed': [], 'Total Moves': []})
training_start = time()
for e in range(episodes):
    state = env.reset()
    # print(env.board)
    # print(env.golden)
    # agent.predict(state, True)
    done = False
    episode_start = time()
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        agent.replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward
        # Only run it for one move so it learns how to match, maybe?
        break

    print(
        f"Episode {e + 1}/{episodes} - Reward: {total_reward} - Epsilon: {agent.epsilon:.3f} - Move Count: {env.total_matches}"
    )
    df = pd.concat([df, pd.DataFrame(
        {'Episode': [e + 1],
         'Reward': [total_reward],
         'Epsilon': [f"{agent.epsilon:.3f}"],
         'Episode Length': [f"{time() - episode_start:.2f}"],
         'Time Elapsed': [f"{time() - training_start:.2f}"],
         'Total Moves': [env.total_matches]}
    )])
    df.to_csv('model_data.csv', index=False)
    agent.replay(batch_size)
    # Update weights every 10 episodes
    if e % 10 == 0:
        agent.update_target_model()
        agent.model.save('uxmal_model.keras')
        pickle.dump(agent.replay_buffer, open('uxmal_replay.pkl', 'wb'))
