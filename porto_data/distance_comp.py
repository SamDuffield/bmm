

import numpy as np
import matplotlib.pyplot as plt

# Load bmm particles
bmm_dir = '/Users/samddd/Main/Data/bayesian-map-matching/simulations/porto/0/0/'
# bmm_particles = np.load(bmm_dir + 'fl_bsi.npy', allow_pickle=True)[2, -1, -1]
# bmm_particles = np.load('/Users/samddd/Desktop/fl_bsi.npy', allow_pickle=True)[0]
bmm_particles = np.load('/Users/samddd/Desktop/ffbsi.npy', allow_pickle=True)[0]

# Load Viterbi particle
viterbi_path = '/Users/samddd/Main/Data/bayesian-map-matching/osrm/'
viterbi_particle = np.load(viterbi_path + 'single_route_matched_particle.npy')


observation_times = bmm_particles.observation_times

# Trim to start of route
n_max = 12
trimmed_observation_times = observation_times[:n_max+1]
max_time = observation_times[n_max]

bmm_particles_trim = bmm_particles
for i in range(bmm_particles.n):
    ind_max_time = np.where(bmm_particles[i][:, 0] == max_time)[0][0]
    bmm_particles_trim.particles[i] = bmm_particles_trim[i][:ind_max_time+1]

ind_max_time_viterbi = np.where(viterbi_particle[:, 0] == max_time)[0][0]
viterbi_particle_trim = viterbi_particle[:ind_max_time_viterbi+1]


# Extract distances
bmm_distances = np.empty((n_max, bmm_particles.n))
for i in range(bmm_particles.n):
    bmm_distances[:, i] = bmm_particles_trim[i][:, -1][bmm_particles_trim[i][:, 0] != 0]
viterbi_distances = viterbi_particle_trim[:, -1][viterbi_particle_trim[:, 0] != 0]


# Box plot

flierprops = dict(marker='.')
medianprops = dict(color='black')

fig, ax = plt.subplots()
ax.boxplot(bmm_distances.T, vert=False, showfliers=False,
           flierprops=flierprops, medianprops=medianprops,
           zorder=0)
ax.invert_yaxis()
ax.set_ylabel('t')
ax.scatter(viterbi_distances[::-1], np.arange(n_max, 0, -1), s=15, zorder=1)
plt.tight_layout()

fig, axes = plt.subplots(n_max, sharex=True)
axes[0].xlim = (0, 165)
for i, d in enumerate(bmm_distances):
    axes[i].hist(d, bins=20, color='purple', alpha=0.5, zorder=0, density=True)
    axes[i].set_yticklabels([])
    axes[i].scatter(viterbi_distances[i], 0, s=50, zorder=1, color='blue')
# plt.tight_layout()



