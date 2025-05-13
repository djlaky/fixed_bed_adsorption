import numpy as np
import pandas as pd

FIM_results = np.genfromtxt('RPB_36_point_FIM_results_flat.csv')

# Grab what i corresponds to
num_pts_pressure = 6
num_pts_RPM = 6

pressure_bounds = (120, 250)
RPM_bounds = (0.01, 0.1)

p_enum = np.linspace(pressure_bounds[0], pressure_bounds[1], num_pts_pressure)
RPM_enum = np.linspace(RPM_bounds[0], RPM_bounds[1], num_pts_RPM)

FIM_names = ['FIM {}'.format(i) for i in range(36)]
index_names = ['Pressure (kPa)', 'Rotational Velocity (RPM)'] + FIM_names

new_frame = pd.DataFrame(index=index_names)

for i in range(36):
    readable_data = []
    readable_data.append(p_enum[i // 6])
    readable_data.append(RPM_enum[i % 6])
    for j in FIM_results[i*36 : (i + 1)*36]:
        readable_data.append(j)
    new_frame[i] = readable_data

new_frame.to_csv('FIM_data_RPB_results.csv', index=True)

example_FIM = new_frame[0].iloc[2:]

print(np.asarray(example_FIM).reshape(6,6))
print(np.log10(np.linalg.det(np.asarray(example_FIM).reshape(6,6))))