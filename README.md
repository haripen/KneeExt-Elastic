# KneeExt-Elastic
A simple Hill-type leg-press simulation for a concentric knee extension in Matlab

*Note:* The Matlab function also runs in Octave either [online](https://octave-online.net) or [locally offline on a macOS](https://wiki.octave.org/Octave_for_macOS) or on [Windows](https://wiki.octave.org/Octave_for_Microsoft_Windows)

INPUT

Properties of the system, defining anthropometry, force-velocity relation, activation dynamics, force-length relation, parallel elastic element, serial elastic element, environment and initial conditions.

OUTPUT

Internal and external forces, velocities, and positions of the leg-press sledge and model muscle, resp., the geometrical ratio and muscle activation as a function of time

Further details are given in the header of KneeExtElastic.m

This is an upgrade to the basic simulation [KneeExt](https://github.com/haripen/KneeExt) now adding length dependencies, serial- and parallel elastic elements to the previous model consisting of activation dynamics, force-velocity relation and a geometrical relatrion.

Subject specific values of anthropometry, muscle activation, and force-velocity relationship as well as external loads can be manipulated to understand the effect of geometrical relations, elasticities, length dependencies and muscle properties differing between individuals.

The model was developed during and is described in my dissertation which is available an ResearchGate(1). 

The model is published in the following article:

 - H. Penasso and S. Thaller, “Determination of individual knee-extensor properties from leg extensions and parameter identification,” Mathematical and Computer Modelling of Dynamical Systems, vol. 23, no. 4, pp. 416–438, 2017. [10.1080/13873954.2017.1336633](https://doi.org/10.1080/13873954.2017.1336633)


(1) [https://www.researchgate.net](https://www.researchgate.net/publication/331022620_The_effect_of_fatigue_on_identified_human_neuromuscular_parameters?_sg=BNKbuuRJLLgRZmKNu-z6yhsGkPmiRjIMTSN24I33oBByhZtHETC9izwlNlh6VWDLC-XjphHwdM0mFshlfLtkltsEzlUuyaPaO31RwERd.GhxlPHaifOkYiphbj6BLqfBAFVA9MzgBKgeAqSYPidEex19RyOQn77mzpNreYvwh1qy2TSBwPaa7IQC4Cf_kQg)
