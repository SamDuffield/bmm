---
title: 'bmm: Bayesian Map-matching'
tags:
  - Python
  - map-matching
  - GPS
authors:
  - name: Samuel Duffield
    orcid: 0000-0002-8656-8734
    affiliation: 1
affiliations:
 - name: University of Cambridge
   index: 1
date: 21 July 2021
bibliography: paper.bib
   
---

# Summary

`bmm` provides probabilistic map-matching with uncertainty quantification, built on top of `OSMnx` [@Boeing2017].
Map-matching is the task of converting a polyline (series of noisy location observations - e.g. GPS data)
and a graph (collection of edges and nodes) into a logical route restricted to the graph.
`bmm` uses particle smoothing methods to produce a collection of particles, each of which representing a
continuous, plausible route.

# Statement of need

Map-matching is a vital task for almost all data driven inference with GPS data.
Map-matching is often non-trivial, i.e. when the graph is dense, the observation noise is significant
and/or the time between observations is large. In these cases there may be multiple routes
that could have feasibly generated the observed polyline and returning a single trajectory is suboptimal.
`bmm` adopts a state-space model approach as described in @Duffield2020
and produces a particle approximation that duly represents probabilistic
uncertainty in both the route taken and the positions at observation times. Additionally, `bmm` offers
support for both offline and online computation.


# Acknowledgements

Samuel Duffield acknowledges support from the EPSRC.

`bmm` is built on top of `OSMnx` [@Boeing2017] - a python package assisting with the retrieval and processing
of OpenStreetMap data [@OpenStreetMap]. Although, `bmm` is applicable to be used on any suitably
labelled Networkx graph [@Hagberg2008].

In addition, `bmm` utilises `numpy` [@Harris2020] and `numba` [@Lam2015] for fast scientific calculations,
`pandas` [@Reback2020] and `geopandas` [@Jordahl2020] for spatial data storage and manipulation
as well as `matplotlib` [@Hunter2007] for visualisation.

# References