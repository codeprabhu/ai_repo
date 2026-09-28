. What parts of the generated code were correct immediately?

The LLM correctly generated the basic structure of the search agent, including the state representation, successor generation, frontier management, and goal-checking logic. The implementation of BFS and A* followed the expected algorithmic structure and required only minor modifications before execution.

2. Did you find any bugs or design problems?

Yes. Some parts of the generated code required verification. Common issues included inefficient data structures, incomplete handling of visited states, and assumptions about the environment representation. These issues did not always produce errors but could affect correctness or performance.

3. How did you discover those problems?

The problems were discovered through systematic testing. I compared the returned paths with the expected solutions, examined edge cases such as blocked goals and unreachable destinations, and compared BFS and A* outputs. Testing revealed behaviours that were not obvious from reading the code alone.

4. Did the LLM use terminology or data structures that you did not understand?

The LLM used standard search terminology such as frontier, explored set, heuristic, priority queue, and path cost. Some implementation details required additional review, but understanding the underlying search concepts made the generated code easier to verify.

5. Did you modify the LLM-generated code?

Yes. I modified portions of the implementation to improve readability, add additional test cases, display intermediate results, and verify heuristic behaviour. These changes helped validate the correctness of the search algorithms.

6. Which tests were most useful?

The most useful tests were comparing BFS and A* on the same map, checking behaviour when no path existed, and experimenting with different heuristic functions. These tests verified both correctness and efficiency.

7. Could you have trusted the program without testing it?

No. Even when code appears reasonable, there is no guarantee that it correctly implements the intended algorithm. Testing is necessary to verify correctness, detect logical errors, and ensure that the program behaves as expected under different conditions.

8. What did you understand about A* that you did not understand before implementing it?

Before implementation, A* seemed similar to BFS with a heuristic. After implementing it, I understood how A* balances the accumulated path cost \(g(n)\) with the estimated remaining cost \(h(n)\) through the evaluation function \(f(n)=g(n)+h(n)\). This allows A* to prioritize more promising paths while still maintaining optimality when the heuristic is admissible.

Final Reflection
1. Why is it important to formulate the search problem before writing the search algorithm?

Formulating the search problem clearly defines the states, actions, goal condition, and cost structure of the environment. Without this formulation, it is impossible to determine what the search algorithm should explore or how success should be measured. A well-defined problem specification ensures that the algorithm solves the correct problem and provides a basis for testing and validation.

2. In what sense is A* an “informed” search algorithm?

A* is considered an informed search algorithm because it uses additional knowledge about the problem through a heuristic function. Unlike uninformed methods such as BFS, A* estimates how far each state is from the goal and uses this information to guide exploration. This often allows it to reach solutions more efficiently while still preserving optimality when an admissible heuristic is used.

3. Why does the choice of heuristic matter?

The heuristic determines how effectively A* is guided toward the goal. A good heuristic reduces the number of states explored and improves efficiency. If the heuristic is weak, A* behaves more like uninformed search. If the heuristic overestimates the true cost, it may lose the guarantee of finding an optimal solution.

4. What did the LLM contribute to the engineering process?

The LLM assisted by generating the initial implementation of BFS and A*, suggesting suitable data structures, and providing explanations of search concepts. This accelerated development and reduced the amount of boilerplate code that needed to be written manually.

5. What could go wrong if an engineer simply accepted LLM-generated code without testing it?

The generated code may contain logical errors, incorrect assumptions, inefficient implementations, or subtle bugs that are not immediately visible. Without testing, these issues could lead to incorrect solutions, poor performance, or invalid conclusions. Testing and validation remain the responsibility of the engineer regardless of how the code was produced.


