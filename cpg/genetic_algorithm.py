import numpy as np
from cpg.bioloid_network import BioloidNetwork
import matplotlib.pyplot as plt
import pandas as pd
import random

class GenetigAlgorithm:
    def __init__(self):

        # how many genomes we evolve at once
        self.population_size = 10

        # which real motors the genomes correspond to
        self.labels = ['right_knee_Z', 'left_knee_Z', 'right_hip_X', 'right_hip_Y', 'right_hip_Z', 'left_hip_X', 'left_hip_Y', 'left_hip_Z', 'right_arm_X', 'right_arm_Y', 'right_arm_Z', 'left_arm_X', 'left_arm_Y', 'left_arm_Z', 'right_foot_X', 'right_foot_Z', 'left_foot_X', 'left_foot_Z']
        # labels = ['right_knee_Z', 'left_knee_Z', 'right_hip_Z', 'left_hip_Z', 'right_arm_Z', 'left_arm_Z', 'right_foot_Z', 'left_foot_Z']

        # these are the amount of variables used in our genome
        self.degrees_of_freedom = len(self.labels)

        # init weights, our actual genome
        self.weights = np.random.rand(self.population_size, self.degrees_of_freedom, self.degrees_of_freedom)*2 - 1

        # using simulation time of one walking cycle of accelerometer data
        self.simulation_time = 1989

        # read the accelerometer data and extract only the joints we are interested in into validation data
        accelerometer_data = pd.read_pickle('../accelerometer/accelerometer_data_cycle.pkl')
        self.validation_data = accelerometer_data[self.labels]

        # generations to evolve
        self.generations = 30

        # tournament size to use in selection
        self.tournament_size = round(self.population_size / 5)


    def init_population(self):
        # declare population
        population = []

        # initialise the population with current weights
        for wghts in self.weights:
            population.append(BioloidNetwork(weights=wghts, simulation_time=self.simulation_time))
        return  population

    def get_ouputs(self, population):
        # get the output results of our population based on current genome. put them in a correctly labeled df for easier analysis
        results = []
        for individual in population:
            individual.simulate_neurons()
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

    def get_permutations(self, wghts):
        # crossovers in *n* of the population between randomly selected individuals
        n = round(len(wghts))
        for m in range(n):
            ind1 = random.randint(0, len(selected) - 1)
            ind2 = random.randint(0, len(selected) - 1)
            w1 = wghts[ind1]
            w2 = wghts[ind2]
            w3 = np.copy(w1)
            w1[:, int(len(w1) / 2):] = w2[:, int(len(w1) / 2):]
            w2[:, int(len(w1) / 2):] = w3[:, int(len(w1) / 2):]
        return wghts

    def get_mutations(self, wghts):
        # mutations in *n* of the population in randomly selected individuals
        n = round(len(wghts))
        for m in range(n):
            ind1 = random.randint(0, len(wghts) - 1)
            tmp_w = wghts[ind1]
            length = len(wghts[0])
            ind2 = random.randint(round(self.degrees_of_freedom / 2), self.degrees_of_freedom - 1)
            tmp_w[:, :ind2] = np.random.rand(length, ind2) * 2 - 1
        return wghts

    def plot_fitness(self, mean_f, max_f):
        # plot fitness
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.title('Mean fitness')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.plot(np.linspace(0, len(mean_f), len(mean_f)), mean_f)
        plt.subplot(1, 2, 2)
        plt.title('Max fitness')
        plt.xlabel('Iterations')
        plt.ylabel('Fitness')
        plt.plot(np.linspace(0, len(max_f), len(max_f)), max_f)
        plt.show()

    def plot_individual(self, individual):
        # plot the output signals
        for i in self.labels:
            plt.plot(np.linspace(0, len(individual[i]), len(individual[i])), individual[i])




# this is where the actual evolution happens
if __name__ == "__main__":
    mean_fitness_over_time = []
    max_fitness_over_time = []

    ga = GenetigAlgorithm()

    for n in range(ga.generations):
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
        elite = ga.weights[ind_elite]

        # update (temporary) weights based on selected individuals
        w_temp = []
        for i in selected:
            w_temp.append(np.copy(ga.weights[i]))

        # make permutations and mutations on the genome
        w_temp = ga.get_permutations(w_temp)
        w_temp = ga.get_mutations(w_temp)

        # input the 1337
        w_temp.append(elite)

        # update weights
        ga.weights = np.copy(w_temp)

        print('Generation: %i' % n)

    # use last generation to visualise results
    population = ga.init_population()
    results = ga.get_ouputs(population)
    fitness = ga.get_fitness(results)

    # find the index of the best individual
    index = fitness.index(max(fitness))

    # get best individual
    winner = results[index]

    # get best weights
    w = ga.weights[index]

    # plot output of individual and fitness over time
    ga.plot_individual(winner)
    ga.plot_fitness(mean_fitness_over_time, max_fitness_over_time)

    # pickle data
    winner.to_pickle('outputs/best_individual_output.pkl')
    np.save('outputs/mean_fitness_over_time', mean_fitness_over_time)
    np.save('outputs/max_fitness_over_time', max_fitness_over_time)
    np.save('outputs/winning_weights', w)
    np.savetxt("outputs/weights.csv", w, delimiter=",")
