from abc import ABC, abstractmethod
from networkx import *
import random
import numpy as np
import math

class Solver(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def solve(self, problem):
        pass


class MySolver(Solver):
    def __init__(self, g=4, beta=2, q0=0.9, tau_0=1,
                 tau_min=0.25, tau_max=5, rho=0.1,
                 perc_retain=0.8, CR = 1, R0=0.33,
                 RMR = 0.3, T0 = 100, TF=60, PMR=0.2,
                 C = 10, max_cycles=50):
        super().__init__()

        self.g=g
        self.beta=beta
        self.q0=q0
        self.tau_0=tau_0
        self.tau_min=tau_min
        self.tau_max=tau_max
        self.rho=rho
        self.perc_retain=perc_retain
        self.CR = CR
        self.R0 = R0
        self.RMR = RMR
        self.T0 = T0
        self.TF = TF
        self.PMR = PMR
        self.C = C
        self.max_cycles = max_cycles

        self.n=self.graph=self.N=self.tau= \
            self.G=self.cycle_count= \
            self.nodes=self.problem= \
            self.chromosomes=self.path_length=\
            self.fitness=self.x=self.y=\
            self.gene_pool=None

    def solve(self, problem):
        min_lengths = []

        self.problem=problem
        self.initialize_variables()
        self.step1()
        pass

        self.step2()
        self.step3()
        while self.cycle_count < self.max_cycles:
            self.step4()
            self.sanity_check()
            self.step5()
            self.step6()
            self.step7()
            if self.cycle_count % self.C == 0 or True:
                self.step8()
            self.cycle_count += 1

            min_length = None
            best_path = None
            for i in range(self.g):
                for chromosome in self.chromosomes[i]:
                    length = self.path_len(chromosome)
                    if min_length == None or length < min_length:
                        min_length = length
                        best_path = chromosome
            min_lengths.append(min_length)

        return best_path, min_length , min_lengths

    def sanity_check(self):
        for group_num in range(self.g):
            for k,chromosome in enumerate(self.chromosomes[group_num]):
                if len(set(chromosome)) < len(chromosome):
                    print(f"detected error : ({group_num},{k}) , cycle_count = {self.cycle_count}")
                    crc = list(chromosome[:])
                    for gene in set(chromosome):
                        crc.remove(gene)
                    print(f"repeated = {list(set(crc))}")
                    print(f"chromosome = {chromosome}")
                    raise Exception("chromosome check failed")

    def step8(self):
        # find group with best solution
        gb = -1
        best_fitness = -1
        for group_num in range(self.g):
            for k in range(len(self.chromosomes[group_num])):
                if self.fitness[group_num][k] > best_fitness:
                    best_fitness = self.fitness[group_num][k]
                    gb = group_num

        for i in range(self.g):
            # choose random group other than current
            rand_group = random.randint(0,self.g-1)
            while rand_group == i:
                rand_group = random.randint(0,self.g-1)

            # update pheromone levels
            for r in list(self.graph.nodes):
                for s in list(self.graph.nodes):
                    R1=random.random()
                    R2=random.random()
                    v = 2*R1*(self.tau[rand_group][r][s] - self.tau[i][r][s]) + \
                        2*R2*(self.tau[gb][r][s] - self.tau[i][r][s])
                    self.tau[i][r][s] += v
                    self.tau[i][r][s] = min(self.tau_max,self.tau[i][r][s])
                    self.tau[i][r][s] = max(self.tau_min,self.tau[i][r][s])

    def step7(self):
        self.fitness = np.zeros(shape=(self.g, self.N))
        for i in range(self.g):
            for k in range(self.N):
                self.fitness[i][k] = self.path_len(self.chromosomes[i][k])

    def step6(self):
        k = 1.38064852e-23
        T = self.T0
        del_T = int((self.T0-self.TF)/31)
        while T > self.TF:
            for group_num in range(self.g):
                # mutate chromosome with probability self.RMR
                if random.random() < self.RMR:
                    # select random chromosome
                    idx = random.randint(0, len(self.chromosomes[group_num]) - 1)
                    S = self.chromosomes[group_num][idx]

                    # create mutated gene
                    i = random.randint(0,len(S)-1)
                    j = random.randint(0,len(S)-1)
                    while i == j:
                        j=random.randint(0, len(S) - 1)
                    Sp = S[:]
                    temp = Sp[i]
                    Sp[i] = Sp[j]
                    Sp[j] = temp

                    del_E = self.energy(Sp) - self.energy(S)
                    if del_E < 0:
                        P = math.exp( - del_E / (k * T))
                        if random.random() < P:
                            self.chromosomes[i][idx] = Sp

                # mutate phermone level with probability self.PMR
                if random.random() < self.PMR:
                    city_1 = random.choice(list(self.graph.nodes))
                    city_2 = random.choice(list(self.graph.nodes))
                    while city_1 == city_2:
                        city_2=random.choice(list(self.graph.nodes))
                    rand_pher = self.tau_min + self.tau_max*random.random()
                    self.tau[group_num][city_1][city_2] = rand_pher
                    self.tau[group_num][city_2][city_1] = rand_pher
            T -= del_T

    def step5(self):
        for i in range(self.g):
            chromosome_1=random.choice(self.chromosomes[i])
            chromosome_2=random.choice(self.chromosomes[i])

            # perform crossover with probability CR
            if random.random() > self.CR:
                continue

            # choose bone-crossover or two point crossover
            child = None
            if random.random() > self.R0:
                child = self.bone_crossover(chromosome_1,chromosome_2,i)
            else:
                child = self.two_point_crossover(chromosome_1,chromosome_2)

            # replace random chromosome with child
            rand_idx = random.randint(0,len(self.chromosomes[i])-1)
            self.chromosomes[i][rand_idx] = child

    def step4(self):
        # find fitness of each path
        if self.fitness is None:
            self.fitness = np.ones(shape=(self.g, self.N))

        # select gene pool
        self.gene_pool = np.zeros(shape=(self.g,self.x+self.y,self.n))
        for i in range(self.g):
            sum_fitness=sum(self.fitness[i])
            for j in range(self.x):
                idx = self.roulette_wheel_select(self.fitness[i])
                self.gene_pool[i][j] = self.chromosomes[i][idx]
            for u in range(self.x,self.x+self.y):
                # idx = p*N + q
                idx = self.roulette_wheel_select(self.fitness.flat)
                p = int(idx / self.N)
                q = int(idx % self.N)
                self.gene_pool[i][u] = self.chromosomes[p][q]

    def step3(self):
        self.chromosomes = self.paths

    def step2(self):
        # best path and corresponding distance
        L_best=np.ones(shape=self.g)*-1
        self.path_lengths = np.zeros(shape=(self.g,self.N))
        edges_best=[set() for _ in range(self.g)]
        for i in range(self.g):
            for k in range(self.N):
                edges = set()
                for j in range(len(self.paths[i][k]) - 1):
                    edge = (self.paths[i][k][j],
                            self.paths[i][k][j + 1])
                    self.path_lengths[i][k] += self.dist(*edge)
                    edges.add(edge)
                if L_best[i] == -1 or L_best[i] > self.path_lengths[i][k]:
                    L_best[i] = self.path_lengths[i][k]
                    edges_best[i] = edges

        # update pheromone levels
        for i in range(self.g):
            for r in list(self.graph.nodes):
                for s in list(self.graph.nodes):
                    del_tau = 0
                    if (r, s) in edges_best[i]:
                        del_tau = 1. / L_best[i]
                    self.tau[i][r][s] = (1-self.rho)*self.tau[i][r][s] \
                                            + self.rho*del_tau
                    self.tau[i][r][s] = min(self.tau_max, self.tau[i][r][s])

    def step1(self):
        # STEP 1.1

        # loop over groups
        for i in range(self.g):

            # loop over ants
            for k in range(self.N):

                # which city out of n total cities in final path
                for j in range(self.n):
                    # current city of this ant
                    r=self.G[i][k]
                    self.paths[i][k][j]=r
                    self.J[i][k].remove(r)

                    q=random.random()

                    print(f"finding next city")

                    # find next city
                    if q < self.q0:
                        best_u, max_val=None, -1
                        for u in self.J[i][k]:
                            val=self.tau[i][r][u] * (self.dist_inv(r, u) ** self.beta)
                            if val > max_val or max_val is None:
                                max_val=val
                                best_u=u
                        if not (best_u is None):
                            self.G[i][k]=best_u
                    else:
                        sum_vals=sum(self.tau[i][r][u] * (self.dist_inv(r, u) ** self.beta) for u in self.J[i][k])
                        sum_probs=0
                        nodes_shuffled=list(self.graph.nodes)
                        random.shuffle(nodes_shuffled)
                        for s in nodes_shuffled:
                            if s in self.J[i][k]:
                                prob=(self.tau[i][r][s] * (self.dist_inv(r, s) ** self.beta)) / sum_vals
                                sum_probs+=prob
                                if sum_probs > self.q0:
                                    self.G[i][k]=s
                                    break
        s=set()
        for city in self.paths[0][0]:
            if city in s:
                print("dup " + str(city))
            else:
                s.add(city)
        assert len(set(self.paths[0][0])) - len(self.paths[0][0]) == 0

        # STEP 1.2
        for i in range(self.g):
            for s in list(self.graph.nodes):
                for r in list(self.graph.nodes):
                    if self.tau[i][r][s] < self.tau_min:
                        self.tau[i][r][s]=self.tau_min
                    else:
                        self.tau[i][r][s]=(1 - self.rho) * self.tau[i][r][s] + self.rho * self.tau_0

    def initialize_variables(self):
        self.graph: networkx.Graph=self.problem.get_graph()
        self.cycle_count=0

        self.T = self.T0

        # number of cities / remove cities if necessary
        self.n=len(set(self.graph.nodes))
        if self.n % self.g != 0:
            r=self.n % self.g
            print(f"r = {r}")
            for k in range(r):
                self.graph.remove_node(list(self.graph.nodes)[-1])
            self.n=len(set(self.graph.nodes))

        # number of ants in each group
        assert self.n % self.g == 0
        self.N=int(self.n / self.g)

        self.x = int(self.N * self.perc_retain)
        self.y = int(self.N * (1-self.perc_retain))
        if self.x + self.y < self.N:
            self.x += 1

        # pheromone level between each city in each group
        self.tau=np.ones(shape=(self.g, self.n + 2, self.n + 2)) * self.tau_0

        # location of each ant in each group
        self.G=np.array([[random.choice(list(self.graph.nodes)) for __ in range(self.N)] for _ in range(self.g)])

        # set of cities not yet visited by kth ant of the ith group
        # adjacent to rth node
        #  J[k][i][r]
        self.J=[[set() for ___ in range(self.n + 2)] for __ in range(self.g)]
        for k in range(self.N):
            for i in range(self.g):
                self.J[i][k]=set(self.graph.nodes)

        # path taken by each ant in each group
        self.paths=np.ones(shape=(self.g, self.N, self.n))

        self.fitness = None

    def path_len(self, path):
        length = 0
        for i in range(len(path)-1):
            length += self.dist(path[i],path[i+1])
        return length

    def two_point_crossover(self,chromosome_1,chromosome_2):
        assert len(chromosome_1) == len(chromosome_2)

        # generate cut points
        loc_0 = random.randint(0,len(chromosome_1))
        loc_1 = random.randint(loc_0,len(chromosome_1))

        child = np.array(
                [*chromosome_1[:loc_0],
                 *chromosome_2[loc_0:loc_1],
                 *chromosome_1[loc_1:]])

        # add any unvisited cities
        unvisited = set(gene for gene in chromosome_1 if not(gene in child))
        seen = set()
        for i in range(len(child)):
            if child[i] in seen:
                child[i] = unvisited.pop()
            else:
                seen.add(child[i])

        assert len(set(child)) - len(child) == 0

        return child

    def bone_crossover(self, chromosome_1, chromosome_2, group_num):
        assert len(chromosome_1) == len(chromosome_2)
        length=len(chromosome_1)

        # find shared subsequences
        seq_pairs_list = []
        seqs = []
        for i in range(length):
            for j in range(length):
                if chromosome_1[i] == chromosome_2[j]:
                    last_point = (i-1 % length,
                                  j-1 % length)
                    added = False
                    for k,seq_pairs in enumerate(seq_pairs_list):
                        if seq_pairs[-1] == last_point:
                            seq_pairs.append((i, j))
                            seqs[k].append(chromosome_1[i])
                            added = True
                    if not added:
                        seq_pairs_list.append([(i,j)])
                        seqs.append([chromosome_1[i]])
        seqs = [seq for seq in seqs if len(seq) > 1]

        # use maximum length sequence as base sequence
        child = []
        if len(seqs) > 0:
            child = max(seqs, key=lambda x: len(x))
            seqs.remove(child)
        else:
            child = [random.choice(chromosome_1)]

        while len(child) < len(chromosome_1):
            # print(f"1 child = {child}")
            # print(f"1 seqs = {seqs}")

            # append any intersection with matching start cities
            finished = False
            while not finished:
                start_gene = child[-1]
                for seq in seqs:
                    if seq[0] == start_gene:
                        # make sure no part of sequence is already in child
                        in_child = False
                        for city in seq:
                            in_child = in_child or city in child

                        if not in_child:
                            child = [*child, *seq[1:]]
                        seqs.remove(seq)
                        finished = False
                        break
                else:
                    finished = True
                    break

            # print(f"2 child = {child}")
            # print(f"2 seqs = {seqs}")

            # select next city as the nearest city in both chromosomes
            # with lowest pheromone
            start_city = int(child[-1])
            neighbors_nc = []
            neighbors = []
            self.get_neighbors(chromosome_1, start_city, child, neighbors_nc,neighbors)
            self.get_neighbors(chromosome_2, start_city, child, neighbors_nc,neighbors)

            # print(f"neighbors = {neighbors}")
            # print(f"neighbors_nc = {neighbors_nc}")

            if len(neighbors_nc) > 0:
                max_pheromone = None
                max_neighbor = None
                for neighbor in neighbors_nc:
                    pheromone = self.tau[group_num][start_city][int(neighbor)]
                    if (max_pheromone is None) or pheromone > max_pheromone:
                        max_pheromone = pheromone
                        max_neighbor = neighbor
                child.append(int(max_neighbor))
                # print(f"adding {max_neighbor}")
            else:
                min_dist = -1
                closest_city = -1
                for city in list(self.graph.nodes):
                    if city in child:
                        continue
                    dist = self.dist(start_city, int(city))
                    if dist < min_dist or min_dist == -1:
                        closest_city = city
                        min_dist = dist
                # print(f"adding {closest_city}")
                child.append(int(closest_city))
            # print(f"3 child = {child}")
            # print(f"3 seqs = {seqs}")

        return child

    def get_neighbors(self, chromosome, start_city, child, neighbors_nc, neighbors):
        length = len(chromosome)
        for i in range(len(chromosome)):
            if chromosome[(i+1) % length] == start_city:
                if not (chromosome[i] in child):
                    neighbors_nc.append(chromosome[i])
                if not (chromosome[(i+2) % length] in child):
                    neighbors_nc.append(chromosome[(i+2) % length])

                neighbors.append(chromosome[i])
                neighbors.append(chromosome[(i+2) % length])

    def roulette_wheel_select(self,fitness):
        sum_fitness = sum(fitness)
        r = random.random()
        cur = 0
        idx = 0
        while cur/sum_fitness < r:
            cur += fitness[idx]
            idx += 1
        idx = min(idx,len(fitness)-1)
        return idx

    def energy(self, chromosome):
        total_dist = 0
        for i in range(len(chromosome)-1):
            total_dist += self.dist(chromosome[i],
                                    chromosome[i+1])
        return 1./float(total_dist)

    def dist(self, a, b):
        return self.problem.wfunc(a, b)
    def dist_inv(self, a, b):
        return 1. / self.problem.wfunc(a, b)


class GreedySolver(Solver):
    def solve(self, problem):
        graph: networkx.Graph=problem.get_graph()
        start=cur=random.choice(list(problem.get_nodes()))
        path=[start]
        total_cost=0
        nnodes=len(set(problem.get_nodes()))
        while len(path) != nnodes:
            best=None
            best_weight=None
            for neighbor in graph.neighbors(cur):
                weight=graph.get_edge_data(cur, neighbor)['weight']
                if (best_weight is None) or (weight < best_weight):
                    if neighbor in path:
                        continue
                    best=neighbor
                    best_weight=weight
            path.append(best)
            total_cost+=best_weight
            cur=best
        return path, total_cost