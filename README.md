# Stochastic Variational Inference in the TrueSkill Model
#### UofT STA414 Winter 2020 A2
#### Instructors: David Duvenaud and Jesse Bettencourt

**Background** 

We'll implement a variant of the TrueSkill model, a player ranking system for competitive
games originally developed for Halo 2. It is a generalization of the Elo rating system in Chess. For the
curious, the original 2007 NIPS paper introducing the trueskill paper can be found here: http://papers.nips.cc/paper/3079-trueskilltm-a-bayesian-skill-rating-system.pdf

This assignment is based on one developed by Carl Rasmussen at Cambridge for his course on probabilistic
machine learning: http://mlg.eng.cam.ac.uk/teaching/4f13/1920/

This repository contains the following files:

* `.gitignore` - tells git to ignore certain kinds of files. This prevents you from submitting the auxiliary files created when producing LaTeX documents.
* `README.md` - the text you are currently reading.
* `A2.pdf` assignment solutions by Andy Tao
* `Project.toml` packages for the Julia environment.
* `A2_src.jl` Julia code providing useful functions.
* `A2_starter.jl` starter code for assignment in Julia.
* `autograd_starter.py` some starter if you would like to use Python with autograd.
* `plots/` directory to store your plots.
* `tennis_data.mat` dataset containing outcomes of tennis games.
