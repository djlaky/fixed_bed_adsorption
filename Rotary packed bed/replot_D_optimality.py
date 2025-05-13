import numpy as np
import matplotlib.pyplot as plt

FIM_results = np.genfromtxt('RPB_36_point_FIM_results_flat.csv')

FIM_results_det = []
pressure_vals = []
RPM_vals = []

# Grab what i corresponds to
num_pts_pressure = 6
num_pts_RPM = 6

pressure_bounds = (120, 250)
RPM_bounds = (0.01, 0.1)

p_enum = np.linspace(pressure_bounds[0], pressure_bounds[1], num_pts_pressure)
RPM_enum = np.linspace(RPM_bounds[0], RPM_bounds[1], num_pts_RPM)

p_enum = ['120', '146', '172', '198', '224', '250']

RPM_enum = ['0.01', '0.028', '0.046', '0.064', '0.082', '0.1']


# Setting some vars
min_det = 0
max_det = 0

min_i = 0
max_i = 0

for i in range(36):
    curr_FIM = FIM_results[i*36 : (i + 1)*36]
    
    curr_det = np.log10(np.linalg.det(curr_FIM.reshape((6, 6))))
    
    FIM_results_det.append(curr_det)
    pressure_vals.append(p_enum[i // 6])
    RPM_vals.append(RPM_enum[i % 6])


def plot_heatmap(data, title, y_label, x_label, colorbar_label):
    # set heatmap x,y ranges
    x_tick_labels = np.int32(np.sort(np.unique(data[:, 0])))
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
    ax.set_yticklabels(y_tick_labels, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_xticks(range(len(x_tick_labels)))
    ax.set_xticklabels(x_tick_labels, fontsize=16, rotation=90)
    ax.set_xlabel(x_label, fontsize=16)
    im = ax.imshow(opt_vals.T, cmap=plt.cm.hot_r)
    ba = plt.colorbar(im)
    ba.set_label(colorbar_label, fontsize=16)
    plt.title(title, fontsize=16)
    plt.tight_layout()


# X and Y axis labels
x_label = "Pressure [kPa]"
y_label = "Rotational Velocity [RPM]"

# Plotting D-optimal heatmap
data_D = np.zeros((len(FIM_results_det), 3))
data_D[:, 0] = pressure_vals
data_D[:, 1] = RPM_vals
data_D[:, 2] = np.asarray(FIM_results_det)

plot_heatmap(data_D, "RPB Sensitivity: D-Optimality", y_label, x_label, "log10(det(FIM))")
plt.savefig('RPB_36_point_D_replot.png', format='png', dpi=450)