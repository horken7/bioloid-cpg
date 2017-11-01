# bioloid-cpg
## Evolving walking gate for a bioloid CPG robot

![screenshot](Report/include/figure/auxiliary/frontpage.png?raw=true "screenshot")

A walking cycle for a humanoid robot can be generated through means of reinforced learning algorithms, such as a genetic algorithm. Using data captured from human movements could potentially lead to a faster convergence of the algortihm along with more stable results. A simulation was set up consisting of a CPG network connected to a genetic algorithm, along with a 3D simulation of a humanoid robot. Accelerometer data from a human walking cycle was inserted as a starting point for the algorithm in order to see if the speed and accuracy of the simulation was improved. The results from the simulation was connected to a physical humanoid robot to evaluate the generated movement pattern. Although the results were found to be unstable, the method does show potential for improvement and refinement.

The directory 'accelerometer' handles data cleaning of raw accelerometer data in an ipython notebook, which can output cleaned data to pickle and csv.

The directory 'cpg' contains a model of a Central Pattern Generator, following the Matsuoka model, connected in a network to model a Bioloid robot. Evolved using a Genethic Algorithm to match the captured accelerometer data. The code is rather well documented, contact the author in case of questions.

The repositories 'presentation' and 'report' were used in the assessment of the course 'Humanoid Robotics', given at Chalmers University of Thecnology, fall of 2017. Further descriptions of the project can be read in the report.