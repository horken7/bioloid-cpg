import numpy as np
from cpg.bioloid_network import BioloidNetwork
import matplotlib.pyplot as plt
import pandas as pd
import random
import copy

class GeneticAlgorithm:
    def __init__(self):
        # if 1, then use entire bioloid as genome, else use only weights
        self.simulation_type = 0

        # how many genomes we evolve at once
        self.population_size = 25

        # which real motors the genomes correspond to
        self.labels = ['right_knee_Z', 'left_knee_Z', 'right_hip_X', 'right_hip_Y', 'right_hip_Z', 'left_hip_X', 'left_hip_Y', 'left_hip_Z', 'right_arm_X', 'right_arm_Y', 'right_arm_Z', 'left_arm_X', 'left_arm_Y', 'left_arm_Z', 'right_foot_X', 'right_foot_Z', 'left_foot_X', 'left_foot_Z']
        # labels = ['right_knee_Z', 'left_knee_Z', 'right_hip_Z', 'left_hip_Z', 'right_arm_Z', 'left_arm_Z', 'right_foot_Z', 'left_foot_Z']

        # these are the amount of variables used in our genome
        self.degrees_of_freedom = len(self.labels)

        # init weights, our actual genome
        self.weights = np.random.rand(self.population_size, self.degrees_of_freedom, self.degrees_of_freedom)*2 - 1
        self.weights[1] = np.load('outputs/3.0/winning_weights_pop25_gen500_type0_withouttonic.npy')
        self.weights[2] = np.load('outputs/3.0/winning_weights_pop25_gen500_type0.npy')
        self.weights[3] = np.load('outputs/3.0/winning_weights_pop25_gen500_type1_with.npy')
        self.weights[4] = np.load('outputs/3.0/winning_weights_pop25_gen1000_type0.npy')

        # using simulation time of one walking cycle of accelerometer data
        self.simulation_time = 1989

        # read the accelerometer data and extract only the joints we are interested in into validation data
        accelerometer_data = pd.read_pickle('../accelerometer/accelerometer_data_cycle.pkl')
        self.validation_data = accelerometer_data[self.labels]

        # generations to evolve
        self.generations = 500

        # tournament size to use in selection
        self.tournament_size = round(self.population_size / 5)


    def init_population(self):
        # declare population
        population = []

        # initialise the population with current weights
        for wghts in self.weights:
            population.append(BioloidNetwork(weights=wghts, simulation_time=self.simulation_time, simulation_type=self.simulation_type))
        return  population

    def get_ouputs(self, population):
        # get the output results of our population based on current genome. put them in a correctly labeled df for easier analysis
        results = []
        for individual in population:
            individual.simulate_neurons(self.validation_data.as_matrix())
            r = individual.get_outputs()
            rdf = pd.DataFrame(r)
            rdf.columns = (self.labels)
            results.append(rdf)
        return results

    def get_fitness(self, results):
        # get fitness, using the mean of the correlation between the two signals. may update!
        # other correlation methors are: plt.xcorr, np.correlate, df.corr, etc.
        fitness = []
        for individual in results:
            correlation = self.validation_data.corrwith(individual)
            corr = np.min(abs(correlation))  # TODO changed here, using min and abs!
            fitness.append(corr)
        return  fitness

    def get_tournament_selection(self, fitness):
        # tournament selection on population-1 to save one space for elitism
        selected = []
        while (len(selected) < len(fitness) - 1):
            tournament = []
            tournament_index = []
            for i in range(self.tournament_size):
                index = random.randint(0, len(fitness) - 1)
                tournament_index.append(index)
                tournament.append(fitness[index])
            selected.append(tournament_index[tournament.index(max(tournament))])  # this is the actual fitness criterion
        return selected

    def get_permutations_weights(self, wghts):
        # crossovers in *n* of the population between randomly selected individuals
        n = round(len(wghts))
        for m in range(n):
            ind1 = random.randint(0, n - 1)
            ind2 = random.randint(0, n - 1)
            w1 = wghts[ind1]
            w2 = wghts[ind2]
            w3 = np.copy(w1)
            w1[:, (int(len(w1) / 2)-1):] = w2[:, (int(len(w1) / 2)-1):]
            w2[:, int(len(w1) / 2):] = w3[:, int(len(w1) / 2):]
        return wghts

    def get_mutations_weights(self, wghts):
        # mutations in *n* of the population in randomly selected individuals
        n = round(len(wghts))
        for m in range(n):
            ind1 = random.randint(0, len(wghts) - 1)
            tmp_w = wghts[ind1]
            length = len(wghts[0])
            ind2 = random.randint(round(self.degrees_of_freedom / 2), self.degrees_of_freedom - 1)
            tmp_w[:, :ind2] = np.random.rand(length, ind2) * 2 - 1
        return wghts

    def get_permutations_population(self, population):
        # crossovers in *n* of the population between randomly selected individuals
        n = round(len(population))
        for m in range(n):
            ind1 = random.randint(0, len(population) - 1)
            ind2 = random.randint(0, len(population) - 1)
            p1 = population[ind1]
            p2 = population[ind2]
            w1 = p1.get_weights()
            w2 = p2.get_weights()
            w3 = copy.deepcopy(w1)
            w1[:, int(len(w1) / 2):] = w2[:, int(len(w1) / 2):]
            w2[:, int(len(w1) / 2):] = w3[:, int(len(w1) / 2):]
            population[ind1].set_weights(w1)
            population[ind1].set_weights(w2)
        return population

    def get_mutations_population(self, population):
        # mutations in *n* of the population in randomly selected individuals
        n = round(len(population))
        for m in range(n):
            ind1 = random.randint(0, len(population) - 1)
            tmp_w = population[ind1].get_weights()
            length = len(tmp_w)
            ind2 = random.randint(round(self.degrees_of_freedom / 2), self.degrees_of_freedom - 1)
            tmp_w[:, :ind2] = np.random.rand(length, ind2) * 2 - 1
            population[ind1].set_weights(tmp_w)
        return population

    def plot_fitness(self, mean_f, max_f):
        # plot fitness
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Mean fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.plot(np.linspace(0, len(mean_f)/100, len(mean_f)), mean_f)
        plt.subplot(1, 2, 2)
        plt.title('Max fitness')
        plt.xlabel('Iterations')
        plt.ylabel('Generation')
        plt.plot(np.linspace(0, len(max_f)/100, len(max_f)), max_f)
        plt.show()

    def plot_individual(self, individual):
        # plot the output signals
        for i in self.labels:
            plt.plot(np.linspace(0, len(individual[i]), len(individual[i])), individual[i])




# this is where the actual evolution happens
if __name__ == "__main__":
    mean_fitness_over_time = []
    max_fitness_over_time = []

    ga = GeneticAlgorithm()

    if(ga.simulation_type==1):
        # init the population for each generation (to start on same conditions)
        population = ga.init_population()

    for n in range(ga.generations):
        if(ga.simulation_type != 1):
            # init the population for each generation (to start on same conditions)
            population = ga.init_population()

        # get the outputs for one generation
        results = ga.get_ouputs(population)

        # calculate the fitness and store historical values
        fitness = ga.get_fitness(results)
        mean_fitness_over_time.append(np.mean(fitness))
        max_fitness_over_time.append(np.max(fitness))

        # get selected individuals on population-1 to save space for one elitism
        selected = ga.get_tournament_selection(fitness)

        # save elitism separately
        ind_elite = np.argmax(fitness)

        # if simulation type 1, use entire bioloid as genome, else use the weights
        if(ga.simulation_type == 1):
            elite = copy.deepcopy(population[ind_elite])

            # update (temporary) population based on selected individuals
            p_temp = []
            for i in selected:
                p_temp.append(copy.deepcopy(population[i]))

            p_temp = ga.get_permutations_population(p_temp)
            p_temp = ga.get_mutations_population(p_temp)

            # input the 1337
            p_temp.append(elite)

            # update population
            population = copy.deepcopy(p_temp)

        else:
            elite = ga.weights[ind_elite]

            # update (temporary) weights based on selected individuals
            w_temp = []
            for i in selected:
                w_temp.append(np.copy(ga.weights[i]))


            # make permutations and mutations on the genome
            w_temp = ga.get_permutations_weights(w_temp)
            w_temp = ga.get_mutations_weights(w_temp)

            # input the 1337
            w_temp.append(elite)

            # update weights
            ga.weights = np.copy(w_temp)

        print('Generation: %i' % (n+1))

    # use last generation to visualise results
    population = ga.init_population()
    results = ga.get_ouputs(population)
    fitness = ga.get_fitness(results)

    # find the index of the best individual
    index = fitness.index(max(fitness))

    # get best individual outputs
    winner = results[index]

    # get all weights
    wghts = ga.weights

    # get best weights
    w = ga.weights[index]

    # pickle data
    label = '_pop25_gen500_type0_with'

    winner.to_pickle('outputs/best_individual_output' + label + '.pkl') # subjective best individual
    np.savetxt("outputs/weights" + label + ".csv", w, delimiter=",") # subjective best individual
    np.save('outputs/winning_weights' + label, w) # subjective best individual

    np.save('outputs/mean_fitness_over_time' + label, mean_fitness_over_time) #
    np.save('outputs/max_fitness_over_time' + label, max_fitness_over_time)
    np.save('outputs/all_weights' + label, wghts)


    # plot output of individual and fitness over time
    ga.plot_individual(winner)
    ga.plot_fitness(mean_fitness_over_time, max_fitness_over_time)