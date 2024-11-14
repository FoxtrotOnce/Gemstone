from PIL import Image
import collections
import numpy as np


class Gem:
    def __init__(self, name, img, confidence=0.92, is_special=False, creates_gold=True):
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


def create_weight_map(golden_grid):
    # Create the weighted weight_map for the grid as well

    # weight_map = np.indices((m, n))[0]
    m, n = golden_grid.shape
    weight_map = np.zeros((m, n))
    # Create a sword-like weight around every non-golden tile:
    # 0 0 1 0 0
    # 0 1 2 1 0
    # 1 2 3 2 1
    # 1 2 2 2 1
    # 1 2 2 2 1...
    for pos in np.argwhere(golden_grid == 0):
        # Generate the tip and blade of the sword
        # Formatted as (weight, y, x)
        sword = [(3, 0, 0), (2, -1, 0), (1, -2, 0), (1, -1, -1), (1, -1, 1)]
        for y in range(m - pos[0]):
            sword.extend([(1, y, -2), (2, y, -1), (2, y, 0), (2, y, 1), (1, y, 2)])

        for weight, y, x in sword:
            newPos = (pos[0] + y, pos[1] + x)
            if 0 <= newPos[0] < m and 0 <= newPos[1] < n:
                weight_map[newPos] = max(weight_map[newPos], weight)

    return weight_map


def calculate_golden(grid, golden_grid, weight_map, gem_pos, swap_to_pos):
    '''
    Takes the grid and golden_grid, then calculates what will happen if gem_pos and swap_to_pos are swapped.
    Gold is the amount of golden tiles created (at minimum) by the move
    Weight is the move's weight based on a number of calculations, listed below:

    - How low the move is as to displace more gems.
    - How close matches are to non-golden spaces

    :return: gold: int, weight: int
    '''
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
                golden_grid[pos] = 1
            weight += weight_map[pos]
            if gems[grid[pos]].is_special:
                weight += 1000
            # Shift down all elements above the match positions
            grid[pos] = 0
            while 0 <= pos[0] - 1 < m and grid[pos[0] - 1, pos[1]] > 0:
                grid[pos] = grid[pos[0] - 1, pos[1]]
                grid[pos[0] - 1, pos[1]] = 0
    if not swappedOnce:
        return -1, -1
    return gold, weight


gems = {
    1: Gem('Aztec Gold Coin', Image.open('gems/item1.png'), 0.8, is_special=True),
    2: Gem('Diamond', Image.open('gems/item2.png')),
    3: Gem('Turquoise Head', Image.open('gems/item3.png')),
    4: Gem('Fire Opal Head', Image.open('gems/item4.png')),
    5: Gem('Emerald', Image.open('gems/item5.png')),
    6: Gem('Red Garnet', Image.open('gems/item6.png')),
    7: Gem('Crow-Face Relic', Image.open('gems/item7.png'), creates_gold=False),
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
    18: Gem('Bloodstone Monkey Relic', Image.open('gems/item18.png'), creates_gold=False),
    19: Gem('African Silver Coin', Image.open('gems/item19.png'), 0.9, is_special=True),
    20: Gem('African Gold Coin', Image.open('gems/item20.png'), 0.9, is_special=True),
    21: Gem('Citrine', Image.open('gems/item21.png')),
    22: Gem('White Pearl', Image.open('gems/item22.png')),
    23: Gem('Quartz Timepiece', Image.open('gems/item23.png'), is_special=True),
    24: Gem('Delicate Zircon Crystals', Image.open('gems/item24.png')),
    25: Gem('Unswappable Fluorite', Image.open('gems/item25.png')),
    26: Gem('Lapis', Image.open('gems/item26.png')),
    27: Gem('Exclusive Iolite', Image.open('gems/item27.png')),
    28: Gem('Aventurine Scarab', Image.open('gems/item28.png')),
    29: Gem('Aquamarine', Image.open('gems/item29.png')),
    30: Gem('Black Pearl', Image.open('gems/item30.png'), 0.95),
}
