from abc import ABC, abstractmethod

import progressbar
from networkx import *
import random
import numpy as np
import math
from collections import Counter

class Solver(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def solve(self, problem):
        pass


class MySolver(Solver):
    def __init__(self, g=4, beta=2, q0=0.9, tau_0=1,
                 tau_min=0.5, tau_max=2, rho=0.1,
                 perc_x=0.8,perc_y=0.2,
                 CR = 1, R0=0.33, RMR = 0.3, T0 = 100, TF=0,
                 PMR=0.2,C = 30, max_cycles=100,
                 GA_generations=100):
        super().__init__()

        self.g=g
        self.beta=beta
        self.q0=q0
        self.tau_0=tau_0
        self.tau_min=tau_min
        self.tau_max=tau_max
        self.rho=rho
        self.perc_x=perc_x
        self.perc_y=perc_y
        self.CR = CR
        self.R0 = R0
        self.RMR = RMR
        self.T0 = T0
        self.TF = TF
        self.PMR = PMR
        self.C = C
        self.max_cycles = max_cycles
        self.GA_generations = GA_generations

        self.n=self.graph=self.N=self.tau= \
            self.G=self.cycle_count= \
            self.nodes=self.problem= \
            self.chromosomes=self.path_length=\
            self.fitness=self.x=self.y=\
            self.gene_pool=None

        self.step1_debug = False
        self.step2_debug = False
        self.step3_debug = False
        self.step4_debug = False

    def print_pher(self):
        np.set_printoptions(precision=2)
        for i in range(self.g):
            print(10*"-" + f" {i} " + 10*"-")
            print("tau = ")
            print(f"{self.tau[i]}")

    def print_chromosome(self):
        np.set_printoptions(precision=1)
        for i in range(self.g):
            print(10*"-" + f" {i} " + 10*"-")
            print("chromosomes = ")
            print(f"{self.chromosomes[i]}")

    def print_gene_pool(self):
        np.set_printoptions(precision=1)
        for i in range(self.g):
            print(10*"-" + f" {i} " + 10*"-")
            print("gene pool = ")
            print(f"{self.gene_pool[i]}")

    def solve(self, problem):
        min_lengths = []

        best_path = None
        min_length = float('inf')

        self.problem=problem
        self.initialize_variables()
        for cycle_count in progressbar.progressbar(range(self.max_cycles)):
            # self.print_pher()
            self.step1()
            self.step2()
            self.step3()
            self.T = self.T0
            for gen_count in range(self.GA_generations):
                if gen_count > 0:
                    self.step4()
                self.step5()
                self.step6()
                self.step7()
                self.T -= self.del_T
            if cycle_count % self.C == 0:
                self.step8()

            best_path_gen = None
            min_length_gen = float('inf')
            for i in range(self.g):
                for chromosome in self.chromosomes[i]:
                    length = self.path_len(chromosome)
                    if length < min_length_gen:
                        min_length_gen = length
                        best_path_gen = chromosome
            min_lengths.append(min_length_gen)

            if min_length_gen < min_length:
                min_length = min_length_gen
                best_path = best_path_gen


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
                    if r >= s:
                        continue
                    R1=random.random()
                    R2=random.random()
                    v = 2*R1*(self.get_pheromone(rand_group,r,s) - self.get_pheromone(i,r,s)) + \
                        2*R2*(self.get_pheromone(gb,r,s) - self.get_pheromone(i,r,s))
                    self.set_pheromone(i,r,s,v + self.get_pheromone(i,r,s))

    def step7(self):
        self.compute_fitness()

    def step6(self):
        # perform Mutation
        for group_num in range(self.g):
            # mutate chromosome with probability self.RMR
            if random.random() < self.RMR:
                # select random chromosome
                idx = random.randint(0, self.N - 1)
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
                if self.energy(Sp) > self.energy(S):
                    P = math.exp( - del_E / (self.k * self.T))
                    if random.random() < P:
                        self.chromosomes[group_num][idx] = Sp
                else:
                    self.chromosomes[group_num][idx]=Sp

            # mutate pheromone level of random edge
            if random.random() < self.PMR:
                city_1 = random.choice(list(self.graph.nodes))
                city_2 = random.choice(list(self.graph.nodes))
                while city_1 == city_2:
                    city_2=random.choice(list(self.graph.nodes))
                rand_pher = self.tau_min + (self.tau_max-self.tau_min)*random.random()
                self.set_pheromone(group_num,city_1,city_2,rand_pher)

    def step5(self):
        # perform Crossover operation
        for i in range(self.g):
            idx_1 = random.randint(0,len(self.chromosomes[i])-1)
            chromosome_1=self.chromosomes[i][idx_1]
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
            self.chromosomes[i][idx_1] = child

    def step4(self):
        # find fitness of each path 
        # if self.fitness is None:
        #     self.compute_fitness()

        # if self.step4_debug:
        #     print(f"Step 4: before = {sum(list(self.fitness[0]))/len(list(self.fitness[0]))}")
            # self.print_chromosome()

        # select gene pool
        self.gene_pool = np.zeros(shape=(self.g,self.x+self.y,self.n))
        for i in range(self.g):
            for j in range(self.x):
                idx = self.roulette_wheel_select(self.fitness[i])
                self.gene_pool[i][j] = self.chromosomes[i][idx]
            for u in range(self.x,self.x+self.y):
                # idx = p*N + q
                idx = self.roulette_wheel_select(self.fitness.flat)
                p = int(idx / self.N)
                q = int(idx % self.N)
                self.gene_pool[i][u] = self.chromosomes[p][q]

        # if self.step4_debug:
        #     print(f"Step 4: fitness frac = {sum(list(self.compute_fitness_gp()[0]))/sum(list(self.fitness[0]))}")

        self.chromosomes = self.gene_pool[:]

    def step3(self):
        self.chromosomes = self.paths[:]
        if self.step3_debug:
            print(f"Step 3:")
            self.print_chromosome()

    def step2(self):
        self.update_pheromones_global()

    def step1(self):
        # STEP 1.1
        # build traveling sequences using transition rules

        # set of cities not yet visited by kth ant of the ith group
        # adjacent to rth node
        #  J[k][i][r]
        self.J=[[set() for ___ in range(self.n + 2)] for __ in range(self.g)]
        for k in range(self.N):
            for i in range(self.g):
                self.J[i][k]=set(self.graph.nodes)

        # location of each ant in each group
        self.G=np.array(
            [[random.choice(list(self.graph.nodes)) for __ in range(self.N)] for _ in range(self.g)])

        # loop over groups
        for i in range(self.g):

            # loop over ants
            for k in range(self.N):

                r=self.G[i][k]
                self.paths[i][k][0]=r

                # which city out of n total cities in final path
                for j in range(1, self.n):

                    # current city of this ant
                    self.J[i][k].remove(r)
                    q=random.random()

                    if self.step1_debug: print(f"Step 1: r = {r}")

                    # find next city
                    next_r = -1
                    if q < self.q0:
                        self.step1_debug: print("Step 1: q<q0")
                        # if False:
                        best_u, max_val=None, -1
                        for u in self.J[i][k]:
                            val=self.get_pheromone(i,r,u) * (self.dist_inv(r, u) ** self.beta)
                            if self.step1_debug: print(f"Step 1: u = {u}, val = {val}")
                            if val > max_val or max_val is None:
                                max_val=val
                                best_u=u
                        next_r = best_u
                    else:
                        if self.step1_debug: print("Step 1: q>=q0")
                        sum_vals=sum(self.get_pheromone(i,r,u) * (self.dist_inv(r, u) ** self.beta) for u in self.J[i][k])
                        sum_probs=0
                        nodes_shuffled=list(self.J[i][k])
                        random.shuffle(nodes_shuffled)
                        rand_num = random.random()
                        if self.step1_debug: print(f"nodes_shuffled = {nodes_shuffled}")
                        for s in nodes_shuffled:
                            prob=(self.get_pheromone(i,r,s) * (self.dist_inv(r, s) ** self.beta)) / sum_vals
                            sum_probs+=prob
                            if self.step1_debug: print(f" u = {s}, prob = {prob}")
                            if sum_probs > rand_num:
                                next_r = s
                                break
                        if next_r == -1:
                            next_r = nodes_shuffled[-1]

                    self.paths[i][k][j] = next_r
                    r = next_r

        assert len(set(self.paths[0][0])) - len(self.paths[0][0]) == 0

        # STEP 1.2
        # Local pheromone update
        self.update_pheromones_local()

    def initialize_variables(self):
        self.graph: networkx.Graph=self.problem.get_graph()

        self.T = self.T0

        # number of cities / remove cities if necessary
        self.n=len(set(self.graph.nodes))

        # number of ants in each group
        self.N=30

        self.x = int(self.N * self.perc_x)
        self.y = int(self.N * self.perc_y)
        if self.x + self.y < self.N:
            self.x += 1

        # pheromone level between each city in each group
        self.tau=np.ones(shape=(self.g, self.n + 1, self.n + 1)) * self.tau_0

        # path taken by each ant in each group
        self.paths=np.ones(shape=(self.g, self.N, self.n))

        self.fitness = None

        # temperature variables
        self.k=1.38064852e-23
        self.T=self.T0
        self.del_T=(self.T0 - self.TF) / self.GA_generations

    def update_pheromones_local(self):
        for i in range(self.g):
            for s in list(self.graph.nodes):
                for r in list(self.graph.nodes):
                    if self.get_pheromone(i,r,s) < self.tau_min:
                        self.set_pheromone(i,r,s,self.tau_min)
                    else:
                        self.set_pheromone(i,r,s,
                                           (1 - self.rho) * self.get_pheromone(i,r,s) +
                                           self.rho * self.tau_0
                                           )

    def update_pheromones_global(self):
        # find path lengths
        L_best=np.ones(shape=self.g) * -1
        self.path_lengths=np.zeros(shape=(self.g, self.N))
        best_path_idx=[-1 for i in range(self.g)]
        for i in range(self.g):
            for k in range(self.N):
                for j in range(self.n):
                    edge=(int(self.paths[i][k][j % self.n]),
                          int(self.paths[i][k][(j + 1) % self.n]))
                    self.path_lengths[i][k]+=self.dist(*edge)
                if L_best[i] == -1 or L_best[i] > self.path_lengths[i][k]:
                    L_best[i]=self.path_lengths[i][k]
                    best_path_idx[i]=k

        if self.step2_debug: print(f"Step 2: before global pheromone update")
        if self.step2_debug: self.print_pher()
        # global pheromone update
        for i in range(self.g):
            if self.step2_debug: print(f"Step 2: group {i} path_lengths = {self.path_lengths[i]}")
            if self.step2_debug: print(f"Step 2: L_group_{i}_best = {L_best[i]}")
            if self.step2_debug: print(f"Step 2: group {i} best_path = {self.paths[i][best_path_idx[i]]}")
            for r in list(self.graph.nodes):
                for s in list(self.graph.nodes):
                    if r >= s:
                        continue
                    self.set_pheromone(i,r,s, (1-self.rho)*self.get_pheromone(i,r,s))
                    self.set_pheromone(i,r,s,min(self.tau_max, self.get_pheromone(i,r,s)))
            for j in range(self.n):
                edge=(int(self.paths[i][best_path_idx[i]][j % self.n]),
                      int(self.paths[i][best_path_idx[i]][(j + 1) % self.n]))
                self.set_pheromone(i,edge[0],edge[1],
                                   self.get_pheromone(i,edge[0],edge[1]) +
                                   1./L_best[i])


        if self.step2_debug: print("Step 2: after global pheromone update")
        if self.step2_debug: self.print_pher()

    def compute_fitness(self):
        self.fitness=np.zeros(shape=(self.g, self.N))
        for i in range(self.g):
            for k in range(self.N):
                self.fitness[i][k]=1./self.path_len(self.chromosomes[i][k])

    def set_pheromone(self,group_num,u,v,new_val):
        n1=max(u, v)
        n2=min(u, v)
        new_val = min(new_val,self.tau_max)
        new_val = max(new_val,self.tau_min)
        self.tau[group_num][n1][n2] = new_val
    def get_pheromone(self,group_num,u,v):
        n1 = max(u, v)
        n2 = min(u, v)
        return self.tau[group_num][n1][n2]

    def path_len(self, path):
        length = 0
        for i in range(self.n):
            length += self.dist(int(path[i % self.n]),
                                int(path[(i+1) % self.n]))
        return length

    def two_point_crossover(self,chromosome_1,chromosome_2):
        assert len(chromosome_1) == len(chromosome_2)

        # generate cut points
        loc_0 = random.randint(0,len(chromosome_1))
        loc_1 = random.randint(loc_0,len(chromosome_1))

        # chromosome = None
        # if random.random() < 0.5:
        #     chromosome = chromosome_1
        # else:
        #     chromosome = chromosome_2

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

        # print(10*"-" + f" {group_num} " + 10*"-")
        # print(f"chromosome_1 = {chromosome_1}")
        # print(f"chromosome_2 = {chromosome_2}")
        # print(f"seqs = {seqs}")

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
                            in_child = in_child or (city in child)

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
            neighbors = []
            self.get_neighbors(chromosome_1, start_city, child, neighbors)
            self.get_neighbors(chromosome_2, start_city, child, neighbors)

            # print(f"neighbors = {neighbors}")

            if len(neighbors) > 0:
                max_pheromone = None
                max_neighbor = None
                for neighbor in neighbors:
                    pheromone = self.get_pheromone(group_num, start_city, int(neighbor))
                    if (max_pheromone is None) or pheromone > max_pheromone:
                        max_pheromone = pheromone
                        max_neighbor = neighbor
                child.append(int(max_neighbor))
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

    def get_neighbors(self, chromosome, start_city, child, neighbors):
        length = len(chromosome)
        for i in range(length):
            if chromosome[(i+1) % length] == start_city:
                if not (chromosome[i] in child) and \
                        not (chromosome[i] in neighbors):
                    neighbors.append(chromosome[i])
                if not (chromosome[(i+2) % length] in child) and \
                        not (chromosome[(i+2) % length] in neighbors):
                    neighbors.append(chromosome[(i+2) % length])

    def roulette_wheel_select(self,fitness):
        sum_fitness = sum(fitness)
        r = random.random()
        cur = 0
        idx = 0
        cur+=fitness[idx]
        while cur/sum_fitness < r:
            idx += 1
            cur+=fitness[idx]
        idx = min(idx,len(fitness)-1)
        return idx

    def energy(self, chromosome):
        total_dist = 0
        for i in range(self.n):
            total_dist += self.dist(int(chromosome[i % self.n]),
                                    int(chromosome[(i+1) % self.n]))
        return float(total_dist)

    def dist(self, a, b):
        return self.problem.wfunc(a, b)
    def dist_inv(self, a, b):
        return 1. / (self.problem.wfunc(a, b)+0.0000001)


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
