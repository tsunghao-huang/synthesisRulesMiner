# Syntehsis Rules Miner
This repository contains a process discovery algorithm that can discover sound free-choice workflow net with non-block structures.
It applies the synthesis rules [[2]](#2) from free-choice nets theory to ensure that the discovered process models are always sound [[1]](#1) and free-choice, which are the desirable properties for workflow nets.

## Demo
The demo.py file demonstrates how to use the algorithm.

The log used for the demo is 

$L = [\langle a,b,c,d,f,g,h\rangle^{22}, \langle a,b,c,f,d,g,h\rangle^{14},\langle a,e,b,c,d,f,g,h\rangle^{13},\langle a,e,b,c,f,d,g,h\rangle^{13},\\\langle a,e,b,c,f,g,d,h\rangle^{10},\langle a,b,c,f,g,d,h\rangle^{10},\langle a,b,e,c,d,f,g,h\rangle^{6},\langle a,b,e,c,f,g,d,h\rangle^{3},\\\langle a,b,e,c,f,d,g,h\rangle^{3},\langle a,b,c,d,e,f,g,h\rangle^{2},\langle a,b,c,e,d,f,g,h\rangle^{2},\langle a,b,c,e,f,g,d,h\rangle^{1},\\\langle a,b,c,e,f,d,g,h\rangle^{1}]$.

The corresponding model used to generate the log is shown as below, which can be rediscovered by the algorithm.
![demo](demo.png)

# Experiments
For replicability of the experiment, the artifacts are provided.
## Data
The event logs used for the experiment can be found in the data folder.
It contains five event logs extracted from four public-available real-life event logs, which are:

- BPI Challenge 2017: https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f
- helpdesk: https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb
- hospitalBilling: https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741
- Road traffic fine  management: https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5

## Experiment 1
The experiment compares the quality of the intermediate models from the existing work, ProDiGy, to the ones generated by our approach.
The intermediate models of the five logs for both approaches are provided in folder [./experiments/experiment_1]().

## Experiment 2
We compare two ordering strategies for adding activities. 
The folder [./experiments/experiment_2]() contains all the intermediate and final models produced by the two ordering strategies.
They are stored as numpy arrays, where each element is a dictionary containing the models and the corresponding quality measures such as fitness, precision, F1, etc.  

## Experiment 3
We compare our approach with the state-of-the-art, Inductive Miner - infrequent (IMF) [[4]](#4). 
The folder [./experiments/experiment_3]() contains the final models produced by IMf and our approach.

# References

<a id="1">[1]</a>
van der Aalst, W.M.P.: The application of Petri nets to workflow management. J.
Circuits Syst. Comput. 8(1), 21–66 (1998)

<a id="2">[2]</a>
Desel, Jorg, and Javier Esparza. Free choice Petri nets. No. 40. Cambridge university press, 1995.

<a id="3">[3]</a>
Dixit, P.M., Buijs, J.C.A.M., van der Aalst, W.M.P.: Prodigy : Human-in-the-loop
process discovery. In: RCIS 2018. pp. 1–12. IEEE (2018)

<a id="4">[4]</a>
Leemans, S.J.J., Fahland, D., van der Aalst, W.M.P.: Scalable process discovery
and conformance checking. Softw. Syst. Model. 17(2), 599–631 (2018)