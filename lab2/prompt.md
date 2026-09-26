Write a well-documented Python program implementing a simple planning agent for a warehouse robot.

The warehouse contains locations A, B and C.

Initial State:
At(Robot,A)
At(Package,A)

Goal:
At(Package,C)

Actions:
Move(X,Y)
PickUp(Package)
Drop(Package)

The program should:

1. Represent states as logical facts.
2. Represent actions using preconditions and effects.
3. Check whether an action is applicable.
4. Apply action effects to generate successor states.
5. Use search to find a valid plan.
6. Print the sequence of actions found.
7. Include comments explaining the implementation.