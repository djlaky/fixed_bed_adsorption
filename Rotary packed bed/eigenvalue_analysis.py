import numpy as np

FIM_results = np.genfromtxt('RPB_36_point_FIM_results_flat.csv')

# Grab what i corresponds to
num_pts_pressure = 6
num_pts_RPM = 6

pressure_bounds = (120, 250)
RPM_bounds = (0.01, 0.1)

p_enum = np.linspace(pressure_bounds[0], pressure_bounds[1], num_pts_pressure)
RPM_enum = np.linspace(RPM_bounds[0], RPM_bounds[1], num_pts_RPM)


# Setting some vars
min_det = 0
max_det = 0

min_i = 0
max_i = 0

for i in range(36):
    curr_FIM = FIM_results[i*36 : (i + 1)*36]
    
    curr_det = np.log10(np.linalg.det(curr_FIM.reshape((6, 6))))
    
    # Set max and min
    if i == 0:
        min_det = curr_det
        max_det = curr_det

    # If less than, set new min
    if curr_det < min_det:
        min_det = curr_det
        min_i = i
    
    # If greater than, set new max
    if curr_det > max_det:
        max_det = curr_det
        max_i = i

print('Max det: {:.3f}'.format(max_det))
print('Eigenvalue analysis: ')
print(np.linalg.eig(FIM_results[max_i*36 : (max_i + 1)*36].reshape((6, 6))))
print('Pressure of {:.2f} kPa and rotational velocity of {:.3f} RPM'.format(p_enum[max_i // 6], RPM_enum[(max_i) % 6]))

print('Min det: {:.3f}'.format(min_det))
print('Eigenvalue analysis: ')
print(np.linalg.eig(FIM_results[min_i*36 : (min_i + 1)*36].reshape((6, 6))))
print('Pressure of {:.2f} kPa and rotational velocity of {:.3f} RPM'.format(p_enum[min_i // 6], RPM_enum[(min_i) % 6]))

print(RPM_enum)

# Plotting some results
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

_1, eigv_best = np.linalg.eig(FIM_results[max_i*36 : (max_i + 1)*36].reshape((6, 6)))
_2, eigv_worst = np.linalg.eig(FIM_results[min_i*36 : (min_i + 1)*36].reshape((6, 6)))

# m.fs.RPB.C1, m.fs.RPB.delH_1, m.fs.RPB.delH_2, m.fs.RPB.delH_3, m.fs.RPB.ads.hgx, m.fs.RPB.des.hgx
pretty_param_names = [r'$C_1$', r'$\Delta H_1$', r'$\Delta H_2$', r'$\Delta H_3$', r'$hgx_{ads}$', r'$hgx_{des}$']
lambda_names = [r'$\lambda_{}$'.format(i+1) for i in range(6)]

best_frame = pd.DataFrame(np.abs(eigv_best), index=pretty_param_names, columns=lambda_names)

worst_frame = pd.DataFrame(np.abs(eigv_worst), index=pretty_param_names, columns=lambda_names)


# Plot the worst frame
sns.heatmap(worst_frame, annot=True, fmt=".2f", cmap=plt.cm.hot_r, annot_kws={"size": 16}, vmin=0, vmax=1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16, rotation=0)
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig('min_det_solution.png', format='png', dpi=450)
plt.clf()
plt.close()

# Plot the best frame
sns.heatmap(best_frame, annot=True, fmt=".2f", cmap=plt.cm.hot_r, annot_kws={"size": 16}, vmin=0, vmax=1)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16, rotation=0)
cbar = plt.gca().collections[0].colorbar
cbar.ax.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig('max_det_solution.png', format='png', dpi=450)\

print("BEST")
print(np.log10(_1))
print("WORST")
print(np.log10(_2))