Environment
Warehouse represented as a 2D grid.

Initial State
Agent position = S

Goal State
Agent position = G

Actions
Up
Down
Left
Right

Each action moves the agent by one grid cell.

Constraints
Cannot move through obstacles (#)
Cannot leave map boundaries
Path Cost
Each move costs 1

Write a Python implementation of a warehouse navigation agent.

Requirements:
- Represent the warehouse as a 2D grid.
- Implement BFS.
- Implement A* search.
- Use Manhattan distance as the heuristic.
- Return the path from S to G.
- Print the explored nodes and path length.
- Handle cases where no path exists.