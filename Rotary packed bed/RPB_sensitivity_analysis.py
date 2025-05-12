from RPB_experiment import RPB_experiment

from pyomo.contrib.doe import DesignOfExperiments

import pyomo.environ as pyo

import numpy as np
import matplotlib.pyplot as plt

import idaes

num_pts_pressure = 6
num_pts_RPM = 6

pressure_bounds = (120, 250)
RPM_bounds = (0.01, 0.1)

# Remove when not testing small set
# num_pts_pressure = 2
# num_pts_RPM = 2

FIM_results = []
data_pressure = []
data_RPM = []

count = 0

# Grid search
for pressure_val in np.linspace(pressure_bounds[0], pressure_bounds[1], num_pts_pressure):
    for RPM_val in np.linspace(RPM_bounds[0], RPM_bounds[1], num_pts_RPM):
        count += 1
        print("=======Iteration Number: {} =======".format(count))
        print("Design variable values for this iteration: (Pressure: {:.2f} kPa, Rotational Velocity: {:.2f} RPM)".format(pressure_val, RPM_val))

        data_pressure.append(pressure_val)
        data_RPM.append(RPM_val)
        
        doe_experiment = RPB_experiment(
            rpm=RPM_val,
            pressure=pressure_val*1e3,  # Convert kPa to Pa
        )
        
        # Create the design of experiments object using our experiment instance from above
        RPB_DoE = DesignOfExperiments(experiment=doe_experiment,
                                      step=1e-3,
                                      scale_constant_value=1,
                                      scale_nominal_param_value=True,
                                      tee=True,)
        
        try:
            FIM = RPB_DoE.compute_FIM(method='kaug')
        except:
            print("\n\nFAILED ITERATION {}\n\n".format(count))
            FIM = np.zeros(6)
        
        np.savetxt('RPB_36_point_FIM_results_flat_{}.csv'.format(count), FIM.flatten(), delimiter=',')

        FIM_results.append(FIM)


# Extract criteria from FIM
def get_FIM_metrics(result):
    eigenvalues, eigenvectors = np.linalg.eig(result)
    min_eig = min(eigenvalues)

    A_opt = np.log10(np.trace(result))
    D_opt = np.log10(np.linalg.det(result))
    E_opt = np.log10(min_eig)
    ME_opt = np.log10(np.linalg.cond(result))

    return A_opt, D_opt, E_opt, ME_opt

FIM_metrics = []

for i in FIM_results:
    FIM_metrics.append(get_FIM_metrics(i))

FIM_metrics_np = np.asarray(FIM_metrics)

# Make heat map
def plot_heatmap(data, title, y_label, x_label, colorbar_label):
    # set heatmap x,y ranges
    x_tick_labels = np.sort(np.unique(data[:, 0]))
    y_tick_labels = np.sort(np.unique(data[:, 1]))

    # optimality-values
    opt_vals = np.asarray(data[:, 2]).reshape(len(x_tick_labels), len(y_tick_labels))
    
    # Plot the colormap
    fig = plt.figure()

    # Plotting options
    ax = fig.add_subplot(111)
    params = {"mathtext.default": "regular"}
    plt.rcParams.update(params)

    # Plotting data
    ax.set_yticks(range(len(y_tick_labels)))
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels)
    ax.set_xlabel(x_label)
    im = ax.imshow(opt_vals.T, cmap=plt.cm.hot_r)
    ba = plt.colorbar(im)
    ba.set_label(colorbar_label)
    plt.title(title)

# X and Y axis labels
x_label = "Pressure [kPa]"
y_label = "Rotational Velocity [RPM]"

# Draw A-optimality figure
data_A = np.zeros((len(FIM_metrics), 3))
data_A[:, 0] = data_pressure
data_A[:, 1] = data_RPM
data_A[:, 2] = FIM_metrics_np[:, 0]

plot_heatmap(data_A, "RPB Sensitivity: A-Optimality", y_label, x_label, "log10(trace(FIM))")
plt.savefig('RPB_16_point_A.png', format='png', dpi=450)

# Draw D-optimality figure
data_D = np.zeros((len(FIM_metrics), 3))
data_D[:, 0] = data_pressure
data_D[:, 1] = data_RPM
data_D[:, 2] = FIM_metrics_np[:, 1]

plot_heatmap(data_D, "RPB Sensitivity: D-Optimality", y_label, x_label, "log10(det(FIM))")
plt.savefig('RPB_16_point_D.png', format='png', dpi=450)

# Draw E-optimality figure
data_E = np.zeros((len(FIM_metrics), 3))
data_E[:, 0] = data_pressure
data_E[:, 1] = data_RPM
data_E[:, 2] = FIM_metrics_np[:, 2]

plot_heatmap(data_E, "RPB Sensitivity: E-Optimality", y_label, x_label, "log10(min-eig(FIM))")
plt.savefig('RPB_16_point_E.png', format='png', dpi=450)

# Draw ME-optimality figure
data_ME = np.zeros((len(FIM_metrics), 3))
data_ME[:, 0] = data_pressure
data_ME[:, 1] = data_RPM
data_ME[:, 2] = FIM_metrics_np[:, 3]

plot_heatmap(data_ME, "RPB Sensitivity: ME-Optimality", y_label, x_label, "log10(cond(FIM))")
plt.savefig('RPB_16_point_ME.png', format='png', dpi=450)

# Save the FIM outputs in a csv file as a flat array
FIM_results_np = np.asarray(FIM_results)
np.savetxt('RPB_36_point_FIM_results_flat.csv', FIM_results_np.flatten(), delimiter=',')