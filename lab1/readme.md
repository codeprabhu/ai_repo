# Artificial Intelligence Laboratory Exercise

## Constructing a Goal-Based Agent using a Large Language Model

### Name: Shrivaths S Prabhu    

### Roll Number: 2024A7PS0582G

---

# Task 1 – Understanding the Problem

## 1. What is the environment?

The environment is a warehouse represented as a two-dimensional grid. The grid contains free spaces through which the vehicle can move, obstacles that cannot be crossed, a starting position (S), and a goal position (G). The warehouse map represents the world in which the autonomous vehicle operates.

## 2. What is the goal of the agent?

The goal of the agent is to determine a collision-free path from the starting position S to the destination G while avoiding all obstacles.

## 3. What actions are available to the agent?

The agent can perform four possible actions:

* Move Up
* Move Down
* Move Left
* Move Right

Each action moves the vehicle by one grid square.

## 4. What information must the agent maintain?

To make decisions, the agent must maintain:

* Its current position
* The goal position
* The warehouse map
* Locations of obstacles
* Previously visited locations
* The path being explored

This information allows the agent to determine which actions are valid and which actions move it closer to the goal.

## 5. Why is this a goal-based agent rather than a simple reflex agent?

This is a goal-based agent because its actions are selected with respect to an explicit goal, namely reaching the destination G. The agent evaluates possible future states and chooses actions that contribute toward achieving the goal.

A simple reflex agent would only react to the current situation using predefined rules without considering future consequences or planning a path.

---

## Think About It

If the warehouse becomes twice as large, the same search strategy can still be used, but the computational cost increases significantly.

Additional difficulties include:

* Increased search time
* Increased memory requirements
* More possible paths to evaluate
* Greater likelihood of dead ends
* Need for more efficient algorithms such as A* for large environments

---

# Task 2 – Designing the Agent

## Environment

The environment is a warehouse grid containing:

* Start position (S)
* Goal position (G)
* Obstacles (#)
* Free cells (.)

## Current State

The current state is the vehicle's current coordinate within the warehouse.

## Goal

Reach the destination G from the starting position S without colliding with obstacles.

## Available Actions

* Up
* Down
* Left
* Right

## Decision-Making Component

The decision-making component is a search algorithm.

For this implementation, Breadth First Search (BFS) is used because:

* It guarantees the shortest path in an unweighted grid.
* It systematically explores all reachable states.
* It is simple and reliable for navigation problems.

---

## Goal-Based Agent Architecture

```text
+----------------------+
|     Environment      |
|   Warehouse Grid     |
+----------+-----------+
           |
           v
+----------------------+
|     Current State    |
| Vehicle Position     |
+----------+-----------+
           |
           v
+----------------------+
|    Goal-Based Agent  |
|      BFS Search      |
+----------+-----------+
           |
           v
+----------------------+
|     Next Action      |
| U / D / L / R        |
+----------+-----------+
           |
           v
+----------------------+
|    Updated State     |
+----------------------+
```

---

# Task 3 – Prompt Engineering

## Prompt Used

Write a well-documented Python program implementing a goal-based agent for the warehouse navigation problem.

The program should:

* Represent the warehouse as a two-dimensional grid.
* Determine a collision-free path from S to G.
* Avoid all obstacles.
* Print either the path found or a suitable message if no path exists.
* Explain the search algorithm chosen and why it is appropriate.

Generate complete executable Python code.

---

# Search Algorithm Used

The program uses Breadth First Search (BFS).

BFS explores neighboring states level by level. Since each movement has equal cost, BFS guarantees that the first path discovered to the goal is the shortest possible path.

For warehouse navigation, BFS is an appropriate choice because:

* The environment is represented as a grid.
* Every move has equal cost.
* A shortest collision-free path is required.

---

# Program Testing

## Test Case 1

Original warehouse map.

Expected Result:
A valid collision-free path is found.

Actual Result:
Path successfully generated.

## Test Case 2

Additional obstacles added to the map.

Expected Result:
Agent should find an alternative route if one exists.

Actual Result:
Agent successfully identified an alternative path.

## Test Case 3

Goal completely blocked by obstacles.

Expected Result:
No path should be found.

Actual Result:
Program displayed:

"No path exists."

---

# Reflection Questions

## 1. Did the LLM generate a working program on the first attempt?

Yes. The generated BFS implementation executed successfully and produced a valid path from the start position to the goal.

## 2. If not, how can you improve your prompt?

If errors occur, the prompt can be improved by:

* Specifying the desired search algorithm.
* Requesting executable code.
* Requesting comments and documentation.
* Asking the LLM to include test cases.

More detailed prompts generally produce more reliable code.

## 3. What search algorithm did the LLM choose?

Breadth First Search (BFS).

## 4. Why do you think the LLM selected this algorithm?

BFS is a standard solution for shortest-path problems in unweighted grids. It is simple to implement, guarantees the shortest path, and is widely used for navigation problems.

---

# Conclusion

This laboratory exercise demonstrated the design and implementation of a goal-based intelligent agent for warehouse navigation. The agent successfully determined a collision-free path from the starting location to the destination while avoiding obstacles. A Large Language Model was used as a software engineering assistant to generate and refine the implementation. The exercise showed that LLMs can accelerate software development, but the generated code must still be tested and validated to ensure correctness.
