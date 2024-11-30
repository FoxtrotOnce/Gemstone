import tensorflow as tf
import numpy as np
import random
import resources as jq
from time import time
from collections import deque
import joblib
import pickle
from time import time


class JewelQuest3Env:
    def __init__(self, grid, active_gems):
        self.board, self.golden = jq.generate_state(grid, active_gems, False)
        self.done = False
        self.moves = []
        m, n = grid.shape
        for pos in np.ndindex(grid.shape):
            if pos[0] + 1 < m:
                self.moves.append((pos, (pos[0] + 1, pos[1])))
            if pos[1] + 1 < n:
                self.moves.append((pos, (pos[0], pos[1] + 1)))

    def reset(self):
        self.board, self.golden = jq.generate_state(grid, active_gems, False)
        self.done = False
        return np.concatenate((self.board, self.golden), axis=None)

    def step(self, move_index):
        swap1, swap2 = self.moves[move_index]
        if self.done or self.board[swap1] < 0 or self.board[swap2] < 0:
            # If the move is invalid, punish the model
            return np.concatenate((self.board, self.golden), axis=None), -10, self.done
        active_gems = np.unique(self.board)
        active_gems = active_gems[active_gems > 0]
        distribution = jq.calculate_distribution(active_gems)
        self.board[swap1], self.board[swap2] = self.board[swap2], self.board[swap1]
        reward = 0
        # Continue matching gem positions while they exist (cascades)
        matched = {'First run'}
        swapped_once = False
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
                    reward += 1
                    self.golden[pos] = 1
                # Shift down all elements above the match positions and add random gems to the top
                while 0 <= pos[0] - 1 < m and self.board[pos[0] - 1, pos[1]] > 0:
                    self.board[pos] = self.board[pos[0] - 1, pos[1]]
                    self.board[pos[0] - 1, pos[1]] = 0
                    pos = (pos[0] - 1, pos[1])
                self.board[pos] = random.choices(active_gems, distribution)[0]
        if not swapped_once:
            # Punish model for making an invalid match
            return np.concatenate((self.board, self.golden), axis=None), -10, self.done
        done = (self.golden == 1).sum() == (self.board > -1).sum()
        return np.concatenate((self.board, self.golden), axis=None), reward, done


class DQNAgent:
    """
    Code copy and pasted from chatgpt. It works well.
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.model = create_dqn_model((input_size,), output_size)
        self.target_model = create_dqn_model((input_size,), output_size)
        self.update_target_model()
        self.replay_buffer = deque(maxlen=20000)
        # Lower gamma is more greedy than a higher gamma (Lower = immediate reward, Higher = equal weight)
        self.gamma = 0.95
        # Epsilon is the exploration probability. Decay is how much it decreases.
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        # Randomly pick between exploration and exploitation (epsilon threshold)
        if random.random() <= self.epsilon:
            # "Explore" a random output
            return random.randint(0, self.output_size - 1)
        # Predict Q-values for the given state, exploit the highest value
        q_values = self.model.predict(state, verbose=0)
        return np.argmax(q_values[0])

    def predict(self, state, show_qvals=False):
        q_values = self.model.predict(state, verbose=0)
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
            target = self.model.predict(state, verbose=0)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state, verbose=0)
                target[0][action] = reward + self.gamma * np.amax(t[0])
            # Train model on new Q-values
            self.model.fit(state, target, epochs=1, verbose=0)

        # Decay the exploration probability (epsilon)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


def create_dqn_model(input_shape, output_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.Normalization(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(output_shape, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model


grid = np.asarray([
    [-1,  0,  0,  0,  0,  0,  0, -1],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [ 0,  0,  0,  0,  0,  0,  0,  0],
    [-1,  0,  0,  0,  0,  0,  0, -1]
])
active_gems = [2, 3, 4, 5]
env = JewelQuest3Env(grid, active_gems)
m, n = grid.shape
# 2mn is the amount of items in the grid + the golden grid (the total input)
# 2mn-m-n is the amount of possible moves (the total output)
agent = DQNAgent(input_size=2 * m * n, output_size=2 * m * n - m - n)
try:
    agent.model = tf.keras.models.load_model('uxmal_model.keras')
except ValueError:
    pass

try:
    agent.replay_buffer = pickle.load(open('uxmal_model_replay_buffer.pkl', 'rb'))
except FileNotFoundError:
    pass

adapt_data = np.asarray([np.concatenate(jq.generate_state(grid, active_gems), axis=None) for _ in range(1000)])
normalization_layer = agent.model.layers[0]
normalization_layer.adapt(adapt_data)

episodes = 500
# Increasing batch_size decreases time
batch_size = 32
for e in range(episodes):
    state = env.reset().reshape(1, -1)
    done = False
    episode_start = time()
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = next_state.reshape(1, -1)
        agent.replay_buffer.append((state, action, reward, next_state, done))
        state = next_state
        if done:
            print(f"Episode {e+1}/{episodes} - Reward: {reward} - Epsilon: {agent.epsilon:.3f} - Time: {time() - episode_start:.2f}s")
    agent.replay(batch_size)
    # Update weights every 10 episodes
    if e % 10 == 0:
        agent.update_target_model()
        agent.model.save('uxmal_model.keras')
        pickle.dump(agent.replay_buffer, open('uxmal_model_replay_buffer.pkl', 'wb'))

