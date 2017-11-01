from cpg.matsuoka_joint import MatsuokaJoint
import numpy as np
import copy

class BioloidNetwork:
    """ Generic algorithm to simulate a bioloid network, more specifically in CPG applications """
    def __init__(self, weights, simulation_time, simulation_type):
        """
        :param weights: square matrix (np.narray) containing the weights between all neurons in the network
        :param simulation_time: the simulation time for one generation
        """
        self.simulation_time = simulation_time
        self.size = len(weights) # how many masuoka neurons there are in the network
        self.weights = weights
        # if simulating with entire Bioloid as genome, use random outputs else use constant
        if(simulation_type==1):
            self.outputs = np.random.rand(self.size) # init the outputs randomly to start the network, then store the last ouput here
        else:
            self.outputs = np.ones(self.size) # init outputs with ones to avoid stocastic behaviour
        self.neurons = [] # where we store the matsuoka neurons

        self.stored_outputs = np.zeros([self.simulation_time, self.size]) # storing the output over time

        self.init_neurons()

    def init_neurons(self):
        """ Init matsuoka neurons """
        for i in range(self.size):
            self.neurons.append(MatsuokaJoint())

    def simulate_neurons(self, input):
        """ Simulate the neurons for 'simulation_time' amount of updates, NOTE: done sequentially, not in parallel. """
        for timestep in range(self.simulation_time):
            tmp_outputs = np.ones(self.size) # use this to make syncronous update
            for index, neuron in enumerate(self.neurons): # TODO remake with matrices
                input1 = self.weights[index] @ self.outputs # inputs given based on some definition.. may change (TODO, which definition?)
                input2 = 0
                tonic = input[timestep][index]
                output = neuron.get_output(input1=input1, input2=input2, timestep=0.01, tonic=tonic) # REMEMBER, use timestep 0.01, not floating timestep!! compare constant to accelerometer timestep
                tmp_outputs[index] = output
                self.stored_outputs[timestep][index] = output
            self.outputs = copy.deepcopy(tmp_outputs)

    def get_outputs(self):
        """ Simple getter """
        return self.stored_outputs

    def get_neurons(self):
        """ Simple getter """
        return self.neurons

    def get_output(self):
        """ Simple getter """
        return self.outputs

    def get_weights(self):
        """ Simple getter """
        return self.weights

    def set_weights(self, w):
        """ Simple setter """
        self.weights = w

    def set_neurons(self, n):
        """ Simple setter """
        self.neurons = n

    def set_output(self, o):
        self.outputs = o