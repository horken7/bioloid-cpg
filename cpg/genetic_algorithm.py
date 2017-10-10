import numpy as np
from cpg.bioloid_network import BioloidNetwork
import matplotlib.pyplot as plt
import pandas as pd
import random

# how many genomes we evolve at once
population_size = 10

# which real motors the genomes correspond to
# labels = ['right_knee_Z', 'left_knee_Z', 'right_hip_X', 'right_hip_Y', 'right_hip_Z', 'left_hip_X', 'left_hip_Y', 'left_hip_Z', 'right_arm_X', 'right_arm_Y', 'right_arm_Z', 'left_arm_X', 'left_arm_Y', 'left_arm_Z', 'right_foot_X', 'right_foot_Z', 'left_foot_X', 'left_foot_Z']
labels = ['right_knee_Z', 'left_knee_Z', 'right_hip_Z', 'left_hip_Z', 'right_arm_Z', 'left_arm_Z', 'right_foot_Z', 'left_foot_Z']

# these are the amount of variables used in our genome
degrees_of_freedom = len(labels)

# init weights, our actual genome
weights = np.random.rand(population_size, degrees_of_freedom, degrees_of_freedom)*2 - 1

# using simulation time of one walking cycle of accelerometer data
simulation_time = 1989

# read the accelerometer data and extract only the joints we are interested in into validation data
accelerometer_data = pd.read_pickle('../accelerometer/accelerometer_data_cycle.pkl')
validation_data = accelerometer_data[labels]

# declare population
population = []

# initialise the population with current weights
for wghts in weights:
    population.append(BioloidNetwork(weights=wghts, simulation_time=simulation_time))



# BELOW IS WHAT IS NEEDED FOR THE EVOLUTION
generations = 10
mean_fitness_over_time = []
max_fitness_over_time = []
tournament_size = 1

for apa in range(generations):

    # get the output results of our population based on current genome. put them in a correctly labeled df for easier analysis
    results = []
    for individual in population:
        individual.simulate_neurons()
        r = individual.get_outputs()
        rdf = pd.DataFrame(r)
        rdf.columns = (labels)
        results.append(rdf)

    # get fitness, using the mean of the correlation between the two signals. may update!
    # other correlation methors are: plt.xcorr, np.correlate, df.corr, etc.
    fitness = []
    for individual in results:
        correlation = validation_data.corrwith(individual)
        mean = np.min(correlation) # TODO changed here
        fitness.append(mean)
    mean_fitness_over_time.append(np.mean(fitness))
    max_fitness_over_time.append(np.max(fitness))

    # first phase of evolution
    if(1): #(apa<(generations/2)):
        # tournament selection on population-1 to save elitism
        selected = []
        while(len(selected) < len(fitness)-1):
            tournament = []
            tournament_index = []
            for i in range(tournament_size):
                index = random.randint(0, len(fitness)-1)
                tournament_index.append(index)
                tournament.append(fitness[index])
            selected.append(tournament_index[ tournament.index(max(tournament)) ]) # this is the actual fitness criterion
        tournament_size = tournament_size + 1 # to go towards convergence

        # save elitism
        ind_1337 = np.argmax(fitness)
        elite = weights[ind_1337]


        # update weights based on selected individuals
        w_temp = []
        for i in selected:
            w_temp.append(weights[i])


        # crossovers in a *quarter* the population between randomly selected individuals
        for bepa in range(round(len(selected)/6)):
            ind1 = random.randint(0, len(selected)-1)
            ind2 = random.randint(0, len(selected) - 1)
            w1 = w_temp[ind1]
            w2 = w_temp[ind2]
            w3 = np.copy(w1)
            w1[:,int(len(w1)/2):] = w2[:,int(len(w1)/2):]
            w2[:, int(len(w1) / 2):] = w3[:, int(len(w1) / 2):]

        # mutations in a *quarter* the population in randomly selected individuals
        for cepa in range(round(len(selected) / 6)):
            ind1 = random.randint(0, len(selected)-1)
            tmp_w = w_temp[ind1]
            length = len(tmp_w)
            ind2 = random.randint(0, len(selected)-1)
            tmp_w[:, :ind2] = np.random.rand(length, ind2)

        # input the 1337
        w_temp.append(elite)
        selected.append(ind_1337)

        # get neuron and output configurations of the selected
        n_temp = []
        o_temp = []
        for s in selected:
            p = population[s]
            n_temp.append(p.get_neurons())
            o_temp.append(p.get_output())

        # update weights
        weights = w_temp

        # update the weights of the population
        for i, p in enumerate(population):
            p.set_weights(weights[i])
            p.set_neurons(n_temp[i])
            p.set_output(o_temp[i])

# TODO måste använda CPG parametrarna från sista steget i föregående generation

    # second phase
    # else:
    #     # naive proportional selection, to implement elitism
    #     amount = []
    #     tmp_fitness = np.asarray(fitness)
    #     tmp_fitness = tmp_fitness+100
    #     tmp_fitness[tmp_fitness<0] = 0
    #     for f in tmp_fitness:
    #         if(f != 0):
    #             a = np.floor((f / sum(tmp_fitness)) * len(tmp_fitness))
    #             amount.append(a)
    #         else:
    #             amount.append(0)
    #
    #     w_temp = []
    #     for i, a in enumerate(amount):
    #         for j in range(int(a)):
    #             w_temp.append(weights[i])
    #
    #     if(len(w_temp) < len(weights)):
    #         times = len(weights) - len(w_temp)
    #         for i in range(times):
    #             ind = np.argmax(fitness)
    #             w_temp.append(weights[ind])
    #
    #     weights = w_temp


    print('Generation: %i' % apa)


# VISUALISE RESULT

# print(fitness_over_time)

# get outputs for final weights
# results = []
# for individual in population:
#     individual.simulate_neurons()
#     r = individual.get_outputs()
#     rdf = pd.DataFrame(r)
#     rdf.columns = (labels)
#     results.append(rdf)
#
# # get fitness corresponding to weights
# fitness = []
# for individual in results:
#     correlation = validation_data.corrwith(individual)
#     mean = np.mean(correlation)
#     fitness.append(mean)
#
# # find the index of the best individual
# index = fitness.index(max(fitness))
#
# # get best individual
# vinare = results[index]
#
# # get best weights
# w = weights[index]
#
# # plot the output signals
# for j in labels:
#     plt.plot( np.linspace(0,len(vinare[j]), len(vinare[j])), vinare[j] )
#
# plot fitness
plt.subplot(1,2,1)
plt.title('Mean fitness')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.plot(np.linspace(0, len(mean_fitness_over_time), len(mean_fitness_over_time)), mean_fitness_over_time)
plt.subplot(1,2,2)
plt.title('Max fitness')
plt.xlabel('Iterations')
plt.ylabel('Fitness')
plt.plot(np.linspace(0, len(max_fitness_over_time), len(max_fitness_over_time)), max_fitness_over_time)
plt.show()
#
# # pickle data
# vinare.to_pickle('outputs/best_individual_output.pkl')
# np.save('outputs/fitness_over_time', fitness_over_time)
# np.save('outputs/winning_weights', w)
# np.savetxt("outputs/weights.csv", w, delimiter=",")
