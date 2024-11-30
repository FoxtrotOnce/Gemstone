from PIL import Image
import pyautogui as pag
import pyscreeze as pyscr
import collections
import numpy as np
from time import time
from timeit import timeit
import random
import heapq


goldSquare = Image.open('Gold Tile.png').resize((47, 47))


class Gem:
    """
    Class for one of the 30 gems in JQ3.
    Stores information about the gem, useful for image recognition and array manipulation
    """
    def __init__(self, name, img, confidence=0.92, is_special=False, creates_gold=True, frequency=1):
        self.name = name
        if name == 'Citrine':
            self.img = img.crop((0, 0, 50, 50))
            self.img_small = img.crop((15, 15, 35, 35))
        else:
            self.img = img.crop((0, 0, 40, 40)).resize((50, 50))
            self.img_small = self.img.crop((15, 15, 35, 35))
        self.confidence = confidence
        self.is_special = is_special
        self.creates_gold = creates_gold
        self.freq = frequency


class MCTreeNode:
    """
    Deprecated, as the actual program uses DQN, not MCTS.
    Monte-carlo Tree Simulation for a given grid and golden grid.
    """
    def __init__(self, base_grid, base_golden, swap1=(0,0), swap2=(0,0), rewards=0):
        self.grid = base_grid
        self.golden = base_golden
        self.move = (swap1, swap2)
        self.children = []

        self.rewards = rewards
        self.visits = 1
        self.C = math.sqrt(2)
        # completed_children and total_children include the parent
        self.completed_children = int((base_golden == 1).sum() == (base_grid != -1).sum())
        self.total_children = 1

    def best_path(self):
        # Preorder traversal to find golden_grid with most tiles
        # best = (gold, string history, original move)
        best = (0, '->', self)

        def search(node, str_history='', og_move=self):
            nonlocal best
            gold = (node.golden == 1).sum()
            if gold > best[0]:
                best = (gold, str_history, og_move)

            for child in node.children:
                if str_history == '':
                    search(child, str_history + '->' + str(child.move), child)
                else:
                    search(child, str_history + '->' + str(child.move), og_move)

        search(self)
        return (best[0], best[2])
        # print('Best Path:')
        # print(best[1][2:])
        # print(best[2].grid)
        # print(best[2].golden)

    def next_move(self):
        # Return the best move (highest avg rewards) to take from here
        # make sure the move also leads towards board completion (at least one child has a completed board)
        best = (-1, self, -1)
        for child in self.children:
            completion_prob = child.completed_children / child.total_children
            avg_reward = child.rewards / child.total_children
            if completion_prob > best[0] or (completion_prob == best[0] and child.rewards > best[2]):
                best = (completion_prob, child, child.rewards)
        if best[0] == -1:
            print('huh')
        return best[:2]

    def step(self, steps):
        m, n = self.grid.shape
        active_gems = np.unique(self.grid)
        active_gems[active_gems == -2] = 1
        active_gems = active_gems[active_gems > 0]
        for _ in range(steps):
            # Selection
            # Pick from each node using UCT algorithm, nodes that have all of their children completed are not picked.
            backprop = [self]
            self.visits += 1
            while (currNode := backprop[-1]).children:
                bestChild = (-1, 'node')
                for child in currNode.children:
                    weight = child.rewards / child.visits \
                             + self.C * math.sqrt(math.log(currNode.visits) / child.visits)
                    if weight > bestChild[0] and not (child.completed_children == child.total_children):
                        bestChild = (weight, child)
                if type(bestChild[1]) is str:
                    print('ended early')
                    return self
                bestChild[1].visits += 1
                backprop.append(bestChild[1])

            # Expansion + Simulation
            total_reward = 0
            total_completions = 0
            for pos, gem in np.ndenumerate(self.grid):
                if gem < 0:
                    continue

                for adj in ((0, 1), (1, 0)):
                    adj = (adj[0] + pos[0], adj[1] + pos[1])
                    if adj[0] < m and adj[1] < n and self.grid[adj] >= 0:
                        currNode = backprop[-1]
                        # Generate 3 children with the same move, use the minimum for MCTS so good cascades don't happen
                        # Rank moves by how much gold they get (reward)
                        gen_grid, gen_golden, gen_gold = -1, -1, 131
                        for _ in range(3):
                            node = mcts_simulate_move(
                                    currNode.grid, currNode.golden, active_gems, pos, adj
                                )
                            if type(node[0]) is int:
                                break

                            # Pick the worst random gems so cascades have as little effect as possible
                            if node[2] < gen_gold:
                                gen_grid, gen_golden, gen_gold = node
                        else:
                            currNode.children.append(
                                MCTreeNode(gen_grid, gen_golden, pos, adj, gen_gold)
                            )
                            total_completions += currNode.children[-1].completed_children
                            total_reward += gen_gold

            # Backpropagation
            total_children = len(backprop[-1].children)
            while backprop:
                parent = backprop.pop()
                parent.rewards += total_reward
                parent.total_children += total_children
                # make sure the parents are also marked as complete if all of their children are
                total_completions += (parent.completed_children + total_completions == parent.total_children - 1)
                parent.completed_children += total_completions
        return self

    def __repr__(self):
        print(f"Move: {self.move} | Rewards: {self.rewards} | CC/TC: {self.completed_children}, {self.total_children}", end='')
        if self.children:
            print('\n[')
            for child in self.children:
                print(child)
            print(']', end='')
        return ''

# grid = np.array([
#     [-1, 0, 0, 0, 0, 0, 0,-1],
#     [ 0, 0, 0, 0, 0, 0, 0, 0],
#     [ 0, 0, 0, 0, 0, 0, 0, 0],
#     [ 0, 0, 0, 0, 0, 0, 0, 0],
#     [ 0, 0, 0, 0, 0, 0, 0, 0],
#     [ 0, 0, 0, 0, 0, 0, 0, 0],
#     [ 0, 0, 0, 0, 0, 0, 0, 0],
#     [-1, 0, 0, 0, 0, 0, 0,-1]
# ])
# goldenGrid = np.array(grid)
# m, n = grid.shape
# activeGems = [2,3,4,5]
# distribution = calculate_distribution(activeGems)

# Create root
# x = 0
# root = MCTreeNode(generate_state(grid, activeGems)[0], goldenGrid).step(70)
# while root.next_move()[0] > 0:
# # print(root.grid)
# # print(root.golden)
# #
#     root.next_move()
#     x += 1
#     root = MCTreeNode(generate_state(grid, activeGems)[0], goldenGrid).step(70)
# print(root)
# print(x)

# root = MCTreeNode(np.asarray([[2,3,2],
#                               [4,2,3],
#                               [3,5,5]]), np.asarray([[0,0,0],[1,1,0],[1,1,1]]))
# print(root)
# root.step(1)
# print()
# print(root)
# root.step(1)
# print()
# print(root)


def calculate_distribution(gems_to_distribute, can_gen_mask=None):
    if can_gen_mask is None:
        can_gen_mask = [True] * len(gems_to_distribute)

    probability_sum = 0
    for i, gem in enumerate(gems_to_distribute):
        if can_gen_mask[i]:
            probability_sum += gems[gem].freq
    # Create a distribution for the gem chances that adds to 1
    distribution = []
    for i, gem in enumerate(gems_to_distribute):
        if can_gen_mask[i]:
            distribution.append(gems[gem].freq / probability_sum)
        else:
            distribution.append(0)
    return distribution


def generate_state(grid, active_gems, return_rand_golden=True):
    """
    generates a random board state given the grid, and gems to generate.
    Runs at ~1s/1000 calls
    """

    # generate_prob is basically just grid but with distribution at every tile
    # it is required to get the chances for gem generation, while setting gems that cannot generate to 0
    generation_prob = np.full((*grid.shape, len(active_gems)), calculate_distribution(active_gems))

    random_grid = np.zeros(grid.shape)
    if return_rand_golden:
        random_golden = np.random.randint(0, 2, grid.shape)
    else:
        random_golden = np.zeros(grid.shape)
    m, n = grid.shape

    for pos, tile in np.ndenumerate(grid):
        if tile < 0:
            random_grid[pos] = tile
            random_golden[pos] = tile
        else:
            gem = random.choices(active_gems, generation_prob[pos])[0]
            random_grid[pos] = gem
            for adj in ((0, 1), (1, 0)):
                left, right = (pos[0] - adj[0], pos[1] - adj[1]), (pos[0] + adj[0], pos[1] + adj[1])
                if 0 <= left[0] and 0 <= left[1] and right[0] < m and right[1] < n\
                        and random_grid[left] == gem:
                    # If a match will be made by distributing another gem below or to the right,
                    # set its probability to 0
                    mask = []
                    for prob_gem, prob in zip(active_gems, generation_prob[right]):
                        mask.append(prob_gem != gem and prob > 0)
                    generation_prob[right] = calculate_distribution(active_gems, mask)
    return random_grid, random_golden


def recognize_golden(golden_grid, bounding_box):
    """
    Modifies golden_grid to match the new grid.
    """
    pag.moveTo(20, 50)
    pag.keyDown('ctrl')
    pag.sleep(0.1)
    s = pag.screenshot(region=bounding_box).convert('RGBA')
    for pos in np.ndindex(*golden_grid.shape):
        try:
            pyscr.locate(goldSquare, s.crop((pos[1] * 50 + 2,
                                             pos[0] * 50 + 2,
                                             pos[1] * 50 + 49,
                                             pos[0] * 50 + 49)), confidence=0.8)
            # s.alpha_composite(Image.alpha_composite(goldSquare.convert('RGBA'), Image.new('RGBA', (47,47), '#0000FF40')), (pos[1] * 50, pos[0] * 50))
            golden_grid[pos] = 1
        except pyscr.ImageNotFoundException:
            pass
    # s.show()
    pag.keyUp('ctrl')


def create_weight_map(golden_grid):
    """
    Creates a weighted map based on golden_grid, using sword weighing.
    Sword weighing is equivalent to abs(y1-y2)+abs(x1-x2), except it uses abs(x1-x2) if y1 < y2.
    Thus, the weight is a diamond, but extending downwards, forming a sword shape.
    1 2 3 4 3 2 1
    2 3 4 5 4 3 2
    3 4 5 5 5 4 3
    3 4 5 5 5 4 3
    3 4 5 5 5 4 3...
    """

    # weight_map = np.indices((m, n))[0]
    m, n = golden_grid.shape
    weight_map = np.zeros((m, n), dtype=int)
    non_golden_tiles = np.argwhere(golden_grid == 0)
    for pos in np.ndindex(m, n):
        max_weight = 0
        for ngPos in non_golden_tiles:
            if pos[0] < ngPos[0]:
                weight = (ngPos[0] - pos[0]) + abs(ngPos[1] - pos[1])
            else:
                weight = abs(ngPos[1] - pos[1])
            max_weight = max(max_weight, 22 - weight)
        weight_map[pos] = max_weight

    return weight_map


def mcts_simulate_move(grid, golden_grid, active_gems, gem_pos, swap_to_pos):
    """
    Very similar to calculate_golden, but returns the board result of the move with no weight.
    Will use this for MCTS until I make things more organized.
    Runs at ~3s/10000 calls
    """
    grid, golden_grid = np.array(grid), np.array(golden_grid)
    m, n = grid.shape
    grid[swap_to_pos], grid[gem_pos] = grid[gem_pos], grid[swap_to_pos]
    swapped_once = False
    gold = 0
    matched = {'First run'}
    distribution = calculate_distribution(active_gems)
    while len(matched) != 0:
        matched = set()
        for pos, gem in np.ndenumerate(grid):
            if gem <= 0:
                continue

            for adj in ((0, 1), (1, 0)):
                adj1, adj2 = (pos[0] + adj[0], pos[1] + adj[1]), (pos[0] + adj[0] * 2, pos[1] + adj[1] * 2)
                in_area = 0 <= adj2[0] < m and 0 <= adj2[1] < n
                if in_area and grid[adj1] == gem and grid[adj2] == gem:
                    for matchPos in (pos, adj1, adj2):
                        matched.add((matchPos, 1))
                        swapped_once = True

        # Sort matched positions by height so gravity doesn't affect anything
        matched = sorted(matched, reverse=True)
        opened_lock = False
        while matched:
            pos, make_gold = matched.pop()
            if golden_grid[pos] == 0 and make_gold:
                gold += 1
                golden_grid[pos] = 1
            # Shift down all elements above the match positions and add random gems to the top
            # Could make this faster by making numpy shift the column down in one move?
            # The only issue is that it needs to stop as soon as it sees the top of something.
            if grid[pos] == 1 and not opened_lock:
                opened_lock = True
                for pos, gem in np.ndenumerate(grid):
                    if gem == -2:
                        matched.append((pos, 0))
                        matched = sorted(matched, reverse=True)
                        break
            while 0 <= pos[0] - 1 < m and grid[pos[0] - 1, pos[1]] > 0:
                grid[pos] = grid[pos[0] - 1, pos[1]]
                grid[pos[0] - 1, pos[1]] = 0
                pos = (pos[0] - 1, pos[1])
            grid[pos] = random.choices(active_gems, distribution)[0]
    if not swapped_once:
        return -1, -1, -1
    return grid, golden_grid, gold


def calculate_golden(grid, golden_grid, weight_map, gem_pos, swap_to_pos):
    """
    Takes the grid and golden_grid, then calculates what will happen if gem_pos and swap_to_pos are swapped.
    Gold is the amount of golden tiles created (at minimum) by the move
    Weight is the move's weight based on the weight_map, and the value of the gems it swaps.

    :return: gold: int, weight: int
    """
    grid, golden_grid = np.array(grid), np.array(golden_grid)
    m, n = grid.shape
    gold = 0
    weight = 0
    grid[swap_to_pos], grid[gem_pos] = grid[gem_pos], grid[swap_to_pos]
    swappedOnce = False
    matched = {'First run'}
    while len(matched) != 0:
        matched = set()
        for pos, gem in np.ndenumerate(grid):
            if gem <= 0:
                continue

            for adj in ((0, 1), (1, 0)):
                adj1, adj2 = (pos[0] + adj[0], pos[1] + adj[1]), (pos[0] + adj[0] * 2, pos[1] + adj[1] * 2)
                inArea = 0 <= adj2[0] < m and 0 <= adj2[1] < n
                if inArea and grid[adj1] == gem and grid[adj2] == gem:
                    for matchPos in (pos, adj1, adj2):
                        matched.add(matchPos)
                        swappedOnce = True

        # Sort matched positions by height so gravity doesn't affect anything
        for pos in sorted(matched):
            # print(gem_pos, swap_to_pos)
            # print(grid)
            if golden_grid[pos] == 0:
                gold += 1
                # Make sure gold is weighed over everything besides special gems
                weight += 25
                golden_grid[pos] = 1
            weight = max(weight, weight_map[pos])
            if gems[grid[pos]].is_special:
                # Weight special gems 10x as heavy as matching gold tiles
                weight += 250
            # Shift down all elements above the match positions
            grid[pos] = 0
            while 0 <= pos[0] - 1 < m and grid[pos[0] - 1, pos[1]] > 0:
                grid[pos] = grid[pos[0] - 1, pos[1]]
                grid[pos[0] - 1, pos[1]] = 0
                pos = (pos[0] - 1, pos[1])
    if not swappedOnce:
        return -1, -1
    return gold, weight


gems = {
    1: Gem('Aztec Gold Coin', Image.open('gems/item1.png'), 0.8, is_special=True, frequency=20),
    2: Gem('Diamond', Image.open('gems/item2.png')),
    3: Gem('Turquoise Head', Image.open('gems/item3.png')),
    4: Gem('Fire Opal Head', Image.open('gems/item4.png')),
    5: Gem('Emerald', Image.open('gems/item5.png')),
    6: Gem('Red Garnet', Image.open('gems/item6.png')),
    7: Gem('Crow-Face Relic', Image.open('gems/item7.png'), creates_gold=False, frequency=6),
    8: Gem('Tiger\'s Eye', Image.open('gems/item8.png'), 0.95),
    9: Gem('Hardwood Mask Relic', Image.open('gems/item9.png')),
    10: Gem('Ruby', Image.open('gems/item10.png'), 0.98),
    11: Gem('Chinese Jade', Image.open('gems/item11.png'), 0.8),
    12: Gem('Gold Nugget', Image.open('gems/item12.png')),
    13: Gem('Onyx Zebra Relic', Image.open('gems/item13.png'), 0.7),
    14: Gem('Amethyst', Image.open('gems/item14.png')),
    15: Gem('"Chicky" Mask', Image.open('gems/item15.png'), 0.8),
    16: Gem('Amber', Image.open('gems/item16.png')),
    17: Gem('Seashell', Image.open('gems/item17.png'), 0.985),
    18: Gem('Bloodstone Monkey Relic', Image.open('gems/item18.png'), creates_gold=False, frequency=10),
    19: Gem('African Silver Coin', Image.open('gems/item19.png'), 0.9, is_special=True, frequency=6),
    20: Gem('African Gold Coin', Image.open('gems/item20.png'), 0.9, is_special=True, frequency=15),
    21: Gem('Citrine', Image.open('gems/item21.png'), frequency=15),
    # In gold quests the frequency is 10, but for silver it's 6
    22: Gem('White Pearl', Image.open('gems/item22.png'), frequency=6),
    # Quartz Timepiece frequency varies based on level, it ranges from 12-15-20
    23: Gem('Quartz Timepiece', Image.open('gems/item23.png'), is_special=True, frequency=15),
    24: Gem('Delicate Zircon Crystals', Image.open('gems/item24.png')),
    25: Gem('Unswappable Fluorite', Image.open('gems/item25.png')),
    26: Gem('Lapis', Image.open('gems/item26.png')),
    27: Gem('Exclusive Iolite', Image.open('gems/item27.png')),
    28: Gem('Aventurine Scarab', Image.open('gems/item28.png')),
    29: Gem('Aquamarine', Image.open('gems/item29.png')),
    30: Gem('Black Pearl', Image.open('gems/item30.png'), 0.95, frequency=20),
}
