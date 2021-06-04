# Solver für das Electric Vehicle Scheduling Problem

Beim [Vehicle Scheduling Problem (Umlaufplanungsproblem)](https://www.wolfgang-bongartz.de/blog/?id=solver-f%C3%BCr-das-electric-vehicle-scheduling-problem) geht es um die möglichst optimale Planung des Einsatzes von Omnibussen in einem Verkehrsbetrieb. Also darum, für jeden einzelnen Omnibus festzulegen, welche Buslinien in welcher Reihenfolge abgefahren werden sollen. Die allermeisten Omnibusse, die im Linienbetrieb unterwegs sind, werden mit Diesel betrieben. Deshalb spielt das Betanken für die Planung keine Rolle, denn die Busse haben eine ausreichend große Reichweite. Bei batterieelektrisch betriebenen Omnibussen schaut das anders aus. Ihre Reichweite ist (noch) zu gering, um einen ganzen Betriebstag durchstehen zu können, weshalb bei der Einsatzplanung nicht nur zu klären ist, wann und wo das Aufladen stattfindet, sondern auch, wo optimalerweise Ladestationen errichtet werden sollten.

In meiner im Sommer 2017 abgeschlossenen Masterarbeit habe ich mich mit der Implementierung eines Solvers für das Electric Vehicle Scheduling Problem in C/C++ und CUDA beschäftigt. Der Solver verwendet die beiden heuristischen Optimierungsverfahren [Ant Colony Optimization](https://www.wolfgang-bongartz.de/blog/?id=ant-colony-optimization) und [Simulated Annealing](https://www.wolfgang-bongartz.de/blog/?id=simulated-annealing). Die Ant Colony Optimization ist sowohl für die CPU als auch für die GPU implementiert. Simulated Annealing ist nur für die CPU implementiert. Die Masterarbeit findet sich im File 'Masterarbeit_WolfgangBongartz.pdf'.

Folgendes wird benötigt, um das Projekt übersetzen zu können:

- Microsoft Visual Studio 2019 (die Community Edition reicht aus)
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)
- [Boost Library](https://www.boost.org/)
