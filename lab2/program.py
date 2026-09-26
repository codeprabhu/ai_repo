from collections import deque

initial_state = frozenset([
    ("AtRobot", "A"),
    ("AtPackage", "A")
])

goal = ("AtPackage", "C")

connections = {
    "A": ["B"],
    "B": ["A", "C"],
    "C": ["B"]
}


def successors(state):
    state = set(state)
    result = []

    robot_loc = None

    for fact in state:
        if fact[0] == "AtRobot":
            robot_loc = fact[1]

    # Move actions
    for nxt in connections[robot_loc]:
        new_state = set(state)

        new_state.remove(("AtRobot", robot_loc))
        new_state.add(("AtRobot", nxt))

        result.append(
            (f"Move({robot_loc},{nxt})",
             frozenset(new_state))
        )

    # Pickup
    if ("AtPackage", robot_loc) in state:
        new_state = set(state)

        new_state.remove(("AtPackage", robot_loc))
        new_state.add(("Holding", "Package"))

        result.append(
            ("PickUp(Package)",
             frozenset(new_state))
        )

    # Drop
    if ("Holding", "Package") in state:
        new_state = set(state)

        new_state.remove(("Holding", "Package"))
        new_state.add(("AtPackage", robot_loc))

        result.append(
            ("Drop(Package)",
             frozenset(new_state))
        )

    return result


queue = deque()
queue.append((initial_state, []))

visited = {initial_state}

while queue:
    state, path = queue.popleft()

    if goal in state:
        print("Plan Found:\n")

        for i, action in enumerate(path, 1):
            print(f"{i}. {action}")

        break

    for action, next_state in successors(state):
        if next_state not in visited:
            visited.add(next_state)
            queue.append(
                (next_state,
                 path + [action])
            )