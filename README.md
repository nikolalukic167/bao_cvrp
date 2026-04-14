# Multi-objective Capacitated Vehicle Routing Problem

The aim of this project is to analyze, design and implement solutions to an
optimization problem using the techniques and mechanisms of bioinspired algorithms taught in
the Bioinspired Algorithms for Optimization (BAO) course.
To do so, students will develop a programming project in Python language in a group using the
PyCharm programming environment, which provides the necessary tools for the development,
testing, documentation and debugging of source code in Python. Other environments can be
used but must be consulted first with the professors.
Instructions for the development of the Project
Each student must select one of the available problems. The problem will be solved using
the techniques learned during the lectures of the course. The solving must include:

- Solving it with at least one evolutionary algorithm and one swarm intelligence
algorithm.
- Using different representations for the problem, comparing and discussing which
one is the better.
If the problem has constraints, testing different constraint handling techniques.
- If the problem is multi-objective, testing different multi-objective algorithms and
metrics.
- Fine-tuning to find the best hyperparameter configuration for each algorithm used,
showing the results of these experiments with all the configurations tested in the
report.
- The evaluation of the problem must be done taking into account that these
methods are stochastic, so more than 30 executions for each algorithm
configuration must be performed for the sake of a statistically significant
comparison.
- The evaluation of the problem must include comparisons in terms of quality and
speed (runtime), as well as showing for the most significant configurations the
convergence and diversity during the iterative process using some graphics.
 -For the comparison of two algorithmic configurations, statistical significance
testing must be performed.
- Using advanced techniques (e.g. parallelism, co-evolution, memetic algorithms,
…) is allowed, but not mandatory.

Apart from developing a Python project to solve the problem, the students will have to
write a report following the scheme provided in the file “Report template.pdf”, explaining
the conceptualization, analysis, design, implementation and experimental results
obtained during the project development. Also, a 10 minutes presentation explaining all of
these parts must be developed and presented during the presentation sessions.
Regulations and evaluation
The project assignment should be carried out taking into account the following rules:
- The project assignment will be carried out in groups. Each group must independently
develop its own project assignment and submit its own project.
- For the development of bioinspired algorithm functionalities needed in the project
assignment, use the library inspyred. 
• The use of any auxiliary library, such as numpy or pandas, is allowed.

## Problem Description
The Capacitated Vehicle Routing Problem (CVRP) is a fundamental logistics and supply
chain optimization problem in which you have a central depot, a fleet of identical delivery
vehicles with a fixed carrying capacity, and a set of geographically dispersed customers,
each requiring a specific demand of goods. The objective is to design a set of closed
delivery routes to service all customers while minimizing the overall transportation cost.
A solution to this problem is a set of distinct routes, where each route defines the exact
sequence of customers assigned to a specific vehicle. For a solution to be valid, every
customer must be visited exactly once by a single vehicle, all routes must start and end at
the central depot (located at 0,0), and the total demand of all customers on any given
route must be less than or equal to the maximum weight/volume capacity of the vehicle.
Variation

In this version of the problem, the students must implement a multi-objective approach.
The objectives to consider are:

- Minimizing total distance
- Balancing the demand covered by each vehicle, i.e. demand covered by each vehicle should be as similar as possible

### Data
100 samples of data can be downloaded from this webpage: https://github.com/PyVRP/Instances/tree/main/CVRP

Each sample in the dataset consists of a file with the format “X-n[T]-k[V].vrp”, where [T] is
the number of tasks and [V] is the number of available vehicles. Each file is text formatted
and contains the capacity for each vehicle, the coordinates (NODE_COORD_SECTION) for
the tasks and the demand (DEMAND_SECTION) of each task.
