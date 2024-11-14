import numpy as np
import pyautogui as pag
import collections
from PIL import Image
import pyscreeze as ps
from resources import *
import math
import joblib
from random import randrange

pag.PAUSE = 0.1
window = pag.getWindowsWithTitle('Jewel Quest III')[0]
window.moveTo(0,0)
select = Image.open('Select Box.png')
goldSquare = Image.open('Gold Tile.png').resize((47, 47))
gridSquare = Image.open('Grid Square.png')
debug = Image.open('Cross Box.png')
try:
    grid = joblib.load('savedBoard.pkl')
    m, n = grid.shape
    goldenGrid = np.zeros(grid.shape)
    for pos, gem in np.ndenumerate(grid):
        if gem == -1:
            goldenGrid[pos] = -1
except FileNotFoundError:
    pass

window.activate()
pag.sleep(1)

# Step 1: Create the grid (Starts on a level overview page)
'''
Find the left-right and top-bottom ranges, then group squares in each 16 steps to get grid elements
'''
try:
    squares = set()
    xRan, yRan = (2000, 0), (2000, 0)
    visualize = pag.screenshot(region=(540,130,220,200))
    for square in pag.locateAll(gridSquare, visualize, confidence=0.8):
        xRan = (min(square.left, xRan[0]), max(square.left, xRan[1]))
        yRan = (min(square.top, yRan[0]), max(square.top, yRan[1]))
        squares.add((square.left, square.top))
    #     visualize.paste(Image.new('RGB', (5,5), "#FF0000"), (square.left, square.top))
    # visualize.show()
    # pag.sleep(240)

    '''
    -1 = No grid is present
     0 = There is a grid tile present (No gem)
    '''
    m, n = math.ceil((yRan[1] - yRan[0]) / 16), math.ceil((xRan[1] - xRan[0]) / 16)
    grid = np.full((m, n), -1)
    goldenGrid = np.full((m, n), -1)
    for i, y in enumerate(range(yRan[0], yRan[1] + 5, 16)):
        for j, x in enumerate(range(xRan[0], xRan[1] + 5, 16)):
            for xStep in range(x - 1, x + 4):
                for yStep in range(y - 1, y + 4):
                    if (xStep, yStep) in squares:
                        grid[i][j] = 0
                        goldenGrid[i][j] = 0
                        break
                if grid[i][j] == 0:
                    break
    print(grid)
    joblib.dump(grid, 'savedBoard.pkl')
    pag.click((800, 700))
    pag.sleep(15)
except ps.ImageNotFoundException:
    pass


# grid = np.full((8,8), 0)
# grid[0][0] = -1
# grid[0][-1] = -1
# grid[-1][-1] = -1
# grid[-1][0] = -1
# l, t = 389, 301

# Step 2: Locate corresponding x, y positions of matrix to in-game grid
'''
Use BFS to find leftmost and topmost indices (Need different approach for disjoint grids though? Mb not)
'''
gridBox = (2000, 2000, 0, 0)
pag.moveTo(20,50)
s = pag.screenshot(region=window.box)
clickable = (300, 300, 50, 50)
# visualize = pag.screenshot(region=window.box).convert('RGBA')
for gemId, gem in gems.items():
    try:
        found = pag.locateAll(gem.img_small, s, confidence=gem.confidence, grayscale=False)
        for box in found:
            # visualize.alpha_composite(Image.alpha_composite(gem.img_small, Image.new('RGBA', (20,20), '#0000FF80')), (box.left, box.top))
            gridBox = (int(min(gridBox[0], box.left - 15)),
                       int(min(gridBox[1], box.top - 15)),
                       int(max(gridBox[2], box.left + 35)),
                       int(max(gridBox[3], box.top + 35)))
            clickable = (box[0] - 15, box[1] - 15, 50, 50)
    except ps.ImageNotFoundException:
        pass
gridBox = (gridBox[0], gridBox[1], gridBox[2] - gridBox[0], gridBox[3] - gridBox[1])
# visualize.show('visualization.png')
# quit()


pag.moveTo(20, 50)
pag.keyDown('ctrl')
pag.sleep(0.5)
s = pag.screenshot(region=gridBox).convert('RGBA')
try:
    for box in pag.locateAll(goldSquare, pag.screenshot(region=gridBox), confidence=0.8):
        # s.alpha_composite(Image.alpha_composite(goldSquare.convert('RGBA'), Image.new('RGBA', (47,47), '#0000FF40')), (box.left, box.top))
        pos = (round((box.top - 3) / 50), round((box.left - 3) / 50))
        goldenGrid[pos] = 1
except (ps.ImageNotFoundException, IndexError) as exc:
    pag.keyUp('ctrl')
    # s.show()
    pass
# s.show()
pag.keyUp('ctrl')


# Step 3: Get gems at every position on the board, then analyze it and make a move
while not all(goldenGrid[grid != -1]):
    pag.moveTo(20, 50)
    s = pag.screenshot(region=gridBox)
    # visualize = pag.screenshot(region=gridBox).convert('RGBA')
    for gemId, gem in gems.items():
        try:
            found = ps.locateAll(gem.img_small, s, confidence=gem.confidence, grayscale=False)
            for box in found:
                # visualize.alpha_composite(Image.alpha_composite(gem.img_small, Image.new('RGBA', (20,20), '#0000FF80')), (box.left, box.top))
                pos = (round((box.top - 15) / 50), round((box.left - 15) / 50))
                grid[pos] = gemId
        except ps.ImageNotFoundException:
            pass

    # Calculate the best move to do
    # bestMove = [nonGoldenCovered, gem_pos, swap_to_pos, weight]

    # Create the weighted weight_map for the grid as well
    weightMap = create_weight_map(goldenGrid)
    bestMove = [-1, (-1, -1), (-1, -1), -1]
    for pos, gem in np.ndenumerate(grid):
        if gem == -1:
            continue

        for adj in ((0, 1), (1, 0)):
            adj = (adj[0] + pos[0], adj[1] + pos[1])
            if 0 <= adj[0] < m and 0 <= adj[1] < n and grid[adj] != -1:
                # Calculate the best move to play based on the amount of gold tiles it gets, and the weight
                gold, weight = calculate_golden(grid, goldenGrid, weightMap, pos, adj)
                if gold > bestMove[0]:
                    bestMove = [gold, pos, adj, weight]
                elif gold == bestMove[0] and weight > bestMove[3]:
                    bestMove = [gold, pos, adj, weight]

    # Finally, actually perform the move in-game, then repeat the entire process.
    print(f"Gives at least {bestMove[0]} gold tiles: Swap {bestMove[1]} to {bestMove[2]}")
    gemPos, swapToPos = bestMove[1], bestMove[2]
    pag.click(gridBox[0] + 25 + gemPos[1] * 50, gridBox[1] + 25 + gemPos[0] * 50)
    pag.click(gridBox[0] + 25 + swapToPos[1] * 50, gridBox[1] + 25 + swapToPos[0] * 50)
    # Wait until it's possible to make a move again (the select box shows up on a clickable tile)
    while True:
        pag.click(clickable[0] + 25, clickable[1] + 25)
        pag.moveTo(20, 50)
        try:
            pag.locateOnScreen(select, region=clickable, confidence=0.8, grayscale=False)
        except pag.ImageNotFoundException:
            pass
        else:
            break
    pag.click(clickable[0] + 25, clickable[1] + 25)
    # Get current golden_grid by holding ctrl and searching for golden_grid tiles
    # The only time this should error is if you make a swap that removes all golden_grid tiles
    goldenGrid = np.zeros((m, n))
    for pos, gem in np.ndenumerate(grid):
        if gem == -1:
            goldenGrid[pos] = 1

    pag.moveTo(20, 50)
    pag.keyDown('ctrl')
    pag.sleep(0.5)
    s = pag.screenshot(region=gridBox).convert('RGBA')
    try:
        for box in pag.locateAll(goldSquare, pag.screenshot(region=gridBox), confidence=0.8):
            # s.alpha_composite(Image.alpha_composite(goldSquare.convert('RGBA'), Image.new('RGBA', (47,47), '#0000FF40')), (box.left, box.top))
            pos = (round((box.top - 3) / 50), round((box.left - 3) / 50))
            goldenGrid[pos] = 1
    except (ps.ImageNotFoundException, IndexError) as exc:
        pag.keyUp('ctrl')
        print('lower it')
        # s.show()
        pass
    # s.show()
    pag.keyUp('ctrl')

# visualize.show()
# print('Completed Grid Analysis')
# print(grid)

# visualize = Image.new('RGBA', (grid.shape[1] * 50, grid.shape[0] * 50), '#555555FF')
# for pos, gemId in np.ndenumerate(grid):
#     if gemId > 0:
#         visualize.alpha_composite(gems[gemId].img, (pos[1] * 50, pos[0] * 50))
# visualize.show()
