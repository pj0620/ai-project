import tsplib95
import matplotlib.pyplot as plt
from solvers import *

# load undircted graph
# problem = tsplib95.load_problem('tsplib/mytest.tsp')
problem = tsplib95.load_problem('tsplib/ch130.tsp')
graph: networkx.Graph = problem.get_graph()

solver = MySolver(g=2)
path, total_cost, history = solver.solve(problem)
print(f"total path cost (my algorithm) = {total_cost}")

plt.plot(history)
plt.xlabel("iteration")
plt.ylabel("minimum path cost")

solver2 = GreedySolver()
path2, total_cost2 = solver2.solve(problem)
print(f"total path cost (greedy) = {total_cost2}")

plt.show()