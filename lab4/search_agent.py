from collections import deque
import heapq

WAREHOUSE = [
    "#####################",
    "#S....#............G#",
    "#.##....##########..#",
    "#....##.............#",
    "#.######.###.#.###..#",
    "#........#..........#",
    "#####################"
]


class SearchAgent:

    def __init__(self, grid):
        self.grid = [list(row) for row in grid]
        self.rows = len(grid)
        self.cols = len(grid[0])

        self.start = None
        self.goal = None

        for r in range(self.rows):
            for c in range(self.cols):

                if self.grid[r][c] == "S":
                    self.start = (r, c)

                elif self.grid[r][c] == "G":
                    self.goal = (r, c)

    def valid(self, r, c):

        if r < 0 or r >= self.rows:
            return False

        if c < 0 or c >= self.cols:
            return False

        return self.grid[r][c] != "#"

    def neighbors(self, pos):

        r, c = pos

        moves = [
            (-1, 0),  # up
            (1, 0),   # down
            (0, -1),  # left
            (0, 1)    # right
        ]

        result = []

        for dr, dc in moves:

            nr = r + dr
            nc = c + dc

            if self.valid(nr, nc):
                result.append((nr, nc))

        return result

    def reconstruct(self, parent, goal):

        path = []

        current = goal

        while current is not None:
            path.append(current)
            current = parent[current]

        path.reverse()
        return path

    def bfs(self):

        queue = deque([self.start])

        visited = {self.start}

        parent = {self.start: None}

        expanded = 0

        while queue:

            current = queue.popleft()

            expanded += 1

            if current == self.goal:

                return (
                    self.reconstruct(parent, current),
                    expanded
                )

            for nxt in self.neighbors(current):

                if nxt not in visited:

                    visited.add(nxt)

                    parent[nxt] = current

                    queue.append(nxt)

        return None, expanded

    def heuristic(self, pos):

        r, c = pos

        gr, gc = self.goal

        return abs(r - gr) + abs(c - gc)

    def astar(self):

        pq = []

        heapq.heappush(
            pq,
            (self.heuristic(self.start), 0, self.start)
        )

        parent = {
            self.start: None
        }

        g_cost = {
            self.start: 0
        }

        expanded = 0

        while pq:

            _, g, current = heapq.heappop(pq)

            expanded += 1

            if current == self.goal:

                return (
                    self.reconstruct(parent, current),
                    expanded
                )

            for nxt in self.neighbors(current):

                new_cost = g + 1

                if (
                    nxt not in g_cost
                    or
                    new_cost < g_cost[nxt]
                ):

                    g_cost[nxt] = new_cost

                    f = (
                        new_cost
                        +
                        self.heuristic(nxt)
                    )

                    heapq.heappush(
                        pq,
                        (f, new_cost, nxt)
                    )

                    parent[nxt] = current

        return None, expanded


def print_result(name, path, expanded):

    print(f"\n{name}")

    if path is None:

        print("No path found")
        print("Nodes Expanded:", expanded)

    else:

        print("Path Length:", len(path) - 1)
        print("Nodes Expanded:", expanded)
        print("Path:")
        print(path)


def main():

    agent = SearchAgent(WAREHOUSE)

    bfs_path, bfs_expanded = agent.bfs()

    astar_path, astar_expanded = agent.astar()

    print_result(
        "BFS",
        bfs_path,
        bfs_expanded
    )

    print_result(
        "A*",
        astar_path,
        astar_expanded
    )

    print("\nComparison")

    print(
        f"BFS expanded {bfs_expanded} nodes"
    )

    print(
        f"A* expanded {astar_expanded} nodes"
    )

    if (
        bfs_path is not None
        and
        astar_path is not None
    ):

        print(
            "BFS Path Length:",
            len(bfs_path) - 1
        )

        print(
            "A* Path Length:",
            len(astar_path) - 1
        )


if __name__ == "__main__":
    main()