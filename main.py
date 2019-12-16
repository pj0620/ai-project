import tsplib95
import matplotlib.pyplot as plt
from solvers import *

# load undircted graph
# problem = tsplib95.load_problem('tsplib/mytest.tsp')
problem = tsplib95.load_problem('tsplib/brg180.tsp')
# problem = tsplib95.load_problem('tsplib/ch130.tsp')
graph: networkx.Graph = problem.get_graph()

# solver = MySolver(g=2,
#                   perc_x=0.5,
#                   perc_y=0.5,
#                   max_cycles=10,
#                   GA_generations=10)
# CR = 1 => 33948
solver = MySolver(max_cycles=100,
                  CR=0.7,
                  g=6)
path, total_cost, history = solver.solve(problem)
print(f"total path cost (my algorithm) = {total_cost}")
print(f"solution \n {path}")

plt.plot(history)
plt.xlabel("iteration")
plt.ylabel("minimum path cost")

solver2 = GreedySolver()
path2, total_cost2 = solver2.solve(problem)
print(f"total path cost (greedy) = {total_cost2}")

plt.show()