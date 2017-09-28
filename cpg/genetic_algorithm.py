import numpy as np
from cpg.bioloid_network import BioloidNetwork
import matplotlib.pyplot as plt
import pandas as pd

# these are the amount of variables used in our genome
degrees_of_freedom = 18

# how many genomes we evolve at once
population_size = 3

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

# get the output results of our population based on current genome. put them in a correctly labeled df for easier analysis
results = []
for individual in population:
    individual.simulate_neurons()
    r = individual.get_outputs()
    rdf = pd.DataFrame(r)
    rdf.columns = (labels)
    results.append(rdf)

# fet fitness, using the mean of the correlation between the two signals. may update!
# other correlation methors are: plt.xcorr, np.correlate, df.corr, etc.
fitness = []
for individual in results:
    correlation = validation_data.corrwith(individual)
    mean = np.mean(correlation)
    fitness.append(mean)


print(fitness)



# plt.plot(apa.index , apa['left_hip_Z'])
# for i in range(10):
#     plt.plot(np.linspace(0,len(a[:, 5]), len(a[:, 5])), a[:, i+5])
#
# plt.show()
