import numpy as np
from cpg.bioloid_network import BioloidNetwork
import matplotlib.pyplot as plt
import pandas as pd
import random

# these are the amount of variables used in our genome
degrees_of_freedom = 18

# how many genomes we evolve at once
population_size = 10

# which real motors the genomes correspond to
labels = ['right_knee_Z', 'left_knee_Z', 'right_hip_X', 'right_hip_Y', 'right_hip_Z', 'left_hip_X', 'left_hip_Y', 'left_hip_Z', 'right_arm_X', 'right_arm_Y', 'right_arm_Z', 'left_arm_X', 'left_arm_Y', 'left_arm_Z', 'right_foot_X', 'right_foot_Z', 'left_foot_X', 'left_foot_Z']

# our actual genome
weights = np.random.rand(population_size, degrees_of_freedom, degrees_of_freedom)*2 - 1

# using simulation time of one walking cycle of accelerometer data
simulation_time = 1989

# where we will store the population
population = []

# populate the population
for w in weights:
    population.append(BioloidNetwork(weights=w, simulation_time=simulation_time))

# read the accelerometer data and extract only the joints we are interested in into validation data
accelerometer_data = pd.read_pickle('../accelerometer/accelerometer_data_cycle.pkl')
validation_data = accelerometer_data[labels]




# BELOW IS WHAT IS NEEDED TO BE DONE FOR EACH GENERATION
generations = 50
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
        mean = np.mean(correlation)
        fitness.append(mean)

    # tournament selection
    tournament_size = 2
    selected = []
    while(len(selected) <= len(fitness)):
        tournament = []
        tournament_index = []
        for i in range(tournament_size):
            index = random.randint(0, len(fitness)-1)
            tournament_index.append(index)
            tournament.append(fitness[index])
        selected.append(tournament_index[ tournament.index(max(tournament)) ])
    tournament_size += 1 # to go towards convergence

    # update weights based on selected individuals
    w_temp = []
    for i in selected:
        w_temp.append(weights[i])
    weights = w_temp

    # crossovers between a quarter of the selected individuals
    for bepa in range(int(len(selected)/4)):
        ind1 = weights[random.randint(0, len(selected)-1)]
        ind2 = weights[random.randint(0, len(selected) - 1)]
        ind1[:,int(len(ind1)/2):] = ind2[:,int(len(ind1)/2):]
        ind2[:, int(len(ind1) / 2):] = ind1[:, int(len(ind1) / 2):]

    # mutations in a quarter of the individuals
    for cepa in range(int(len(selected) / 4)):
        ind1 = random.randint(0, len(weights)-1)
        tmp_w = weights[ind1]
        length = len(tmp_w)
        ind2 = random.randint(0, len(tmp_w)-1)
        tmp_w[:, :ind2] = np.random.rand(length, ind2)

    print('Generation: %i' % apa)



# VISUALISE RESULT

# get outputs for final weights
results = []
for individual in population:
    individual.simulate_neurons()
    r = individual.get_outputs()
    rdf = pd.DataFrame(r)
    rdf.columns = (labels)
    results.append(rdf)

# get fitness corresponding to weights
fitness = []
for individual in results:
    correlation = validation_data.corrwith(individual)
    mean = np.mean(correlation)
    fitness.append(mean)

# find the index of the best individual
index = fitness.index(max(fitness))

# get best individual
vinare = results[index]

# plot the output signals
for j in labels:
    plt.plot( np.linspace(0,len(vinare[j]), len(vinare[j])), vinare[j] )
plt.show()
