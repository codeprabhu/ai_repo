from collections import deque

warehouse = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################"
]

rows = len(warehouse)
cols = len(warehouse[0])

start = None
goal = None

for r in range(rows):
    for c in range(cols):
        if warehouse[r][c] == 'S':
            start = (r, c)
        elif warehouse[r][c] == 'G':
            goal = (r, c)

directions = [(-1,0),(1,0),(0,-1),(0,1)]

queue = deque([start])
visited = {start}
parent = {}

found = False

while queue:
    current = queue.popleft()

    if current == goal:
        found = True
        break

    for dr, dc in directions:
        nr = current[0] + dr
        nc = current[1] + dc

        if (0 <= nr < rows and
            0 <= nc < cols and
            warehouse[nr][nc] != '#' and
            (nr, nc) not in visited):

            visited.add((nr, nc))
            parent[(nr, nc)] = current
            queue.append((nr, nc))

if found:
    path = []
    node = goal

    while node != start:
        path.append(node)
        node = parent[node]

    path.append(start)
    path.reverse()

    print("Path found:")
    for step in path:
        print(step)

else:
    print("No path exists.")