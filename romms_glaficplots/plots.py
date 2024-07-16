import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')


# Position and Magnification Plots

def error_plot(filename_1, filename_2, filename_4, filename_5, plot_name, num_images):
    # Storage for parsed data
    data = []
    
    val = pd.read_csv(filename_4)
    val.__dataframe__
    val_column = val.columns[0]

    # Split the values in the data_column and expand into separate columns
    val = val[val_column].str.split(expand=True)

    # Convert the DataFrame to numeric type
    val = val.apply(pd.to_numeric)

    # Line by line read (Remove # from obs file)
    with open(filename_1, 'r') as file:
        for line in file:
            # Skip lines starting with "#"
            if line.startswith("#"):
                continue
            
            # Split the line by whitespace
            line_data = line.split()
            
            # Remove # 
            line_data = [float(val) for val in line_data if val != '#']  
            
            data.append(line_data)
    
    # Convert to DataFrame
    data_df = pd.DataFrame(data)

    # Exclude the first row
    data_df = data_df.iloc[1:]

    data_df.insert(8, "Label", ['Red Image','Green Image','Yellow Image','Blue Image'], True)

    data_df = data_df.drop(columns =[3, 5, 6, 7])

    # Read and process the predicted data
    data_pred = pd.read_csv(filename_2, header=None, delim_whitespace=True, comment='#')
    df_pred = data_pred.iloc[1:]

    # Function for swapping data 
    def swap_rows(df, row1, row2):
        df.iloc[row1], df.iloc[row2] =  df.iloc[row2].copy(), df.iloc[row1].copy()
        return df
    
    # For loop to iterate over row range for row swapping
    for i in range(4):
        diff = abs(abs(data_df.iloc[i,0]) - abs(df_pred[0]))
        m = diff.idxmin()
        n = min(diff)
        if n < 0.01:
            df_pred = swap_rows(df_pred, i, (m-1))
        else:
            continue
        
    df_pred = df_pred.drop(columns =[3])

    # Eliminating the 5th image
    if len(df_pred[2])>num_images:
        i = len(df_pred[2]) - num_images
        for j in range(i):
            min_vales = np.min(abs(df_pred[2]))
            df_2 = abs(df_pred)
            b = df_2.index.get_loc(df_2[df_2[2] == min_vales].index[0])
            df_3 = df_pred.drop((b+1), axis='index')
            df_pred = df_3

    # Calculations for Position Error values
    d_x = data_df[0]-df_pred[0]
    d_y = data_df[1]-df_pred[1]
    sum_sq = (d_x**2) + (d_y**2)
    sq = np.sqrt(sum_sq)
    rms = np.average(sq)
    rms_unit = rms*1000
    rms_round = round(rms_unit, 3)
    rms_str = str(rms_round)

    # Plotting Position Error Graph
    # custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    # sns.set_theme(style="ticks", rc=custom_params)

    colours1 = ['lightsalmon', 'green', 'gold', 'blue']
    colours = [(0.1,0.4,1.0,0.7), (0.0,0.7,0.4,0.7), (0.7,0.3,0.4,0.7), (0.8,0.7,0.1,0.7)]
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.bar(data_df['Label'], sq, color = colours1, width=0.3)
    plt.axhline(xmin=0.045, xmax=0.13, y=0.01, linestyle ='--', color ='r', linewidth = 2, label='1 σ Error')
    plt.axhline(xmin=0.32, xmax=0.40, y=0.01, linestyle ='--', color ='r', linewidth = 2)
    plt.axhline(xmin=0.6, xmax=0.68, y=0.01, linestyle ='--', color ='r', linewidth = 2)
    plt.axhline(xmin=0.87, xmax=0.95, y=0.01, linestyle ='--', color ='r', linewidth = 2)
    plt.title('Position Error, ' + '$Δ^{RMS} _{Pos}$ = ' + rms_str + ' mas')
    plt.legend(loc='upper right', fontsize='small')
    plt.xticks(fontsize=8)
    plt.ylim(0, max(sq)+0.01)
    plt.xlabel('Position error')
    plt.ylabel('Positional offset')

    # Calculations for Magnification value
    f = df_pred[2][1]
    flux = df_pred[2]/f
    df_pred[3] = flux 

    df_pred[3] = abs(df_pred[3])

    # Create data 
    x = np.arange(4)
    width = 0.3

    # FITS image processing for predicted flux at observed positions 
    image = fits.open(filename_5)
    values = image[0].data
    image.close()
    dat = values[6]
    g = data_df[0]*100
    h = data_df[1]*100
    g_max = (data_df[0]+0.01)*100
    h_max = (data_df[1]+0.01)*100
    g_min = (data_df[0]-0.01)*100
    h_min = (data_df[1]-0.01)*100

    x_pos = 350 + g
    y_pos = 350 + h
    x_pos_max = 350 + g_max
    y_pos_max = 350 + h_max
    x_pos_min = 350 + g_min
    y_pos_min = 350 + h_min


    x_pos = x_pos.astype(int)
    y_pos = y_pos.astype(int)
    x_pos_max = x_pos_max.astype(int)
    y_pos_max = y_pos_max.astype(int)
    x_pos_min = x_pos_min.astype(int)
    y_pos_min = y_pos_min.astype(int)

    flux_pos = []
    flux_pos_max = []
    flux_pos_min = []

    for i in range(1,5):
        flux_cal = dat[y_pos[i]][x_pos[i]]
        flux_pos.append(flux_cal)

    for i in range(1,5):
        flux_cal_max = dat[y_pos_max[i]][x_pos_max[i]]
        flux_pos_max.append(flux_cal_max)
    
    for i in range(1,5):
        flux_cal_min = dat[y_pos_min[i]][x_pos_min[i]]
        flux_pos_min.append(flux_cal_min)
    
    
    l = flux_pos[0]
    true_flux = l/flux_pos
    true_flux = abs(true_flux)

    l_max = flux_pos_max[0]
    true_flux_max = l_max/flux_pos_max
    true_flux_max = abs(true_flux_max)

    l_min = flux_pos_min[0]
    true_flux_min = l_min/flux_pos_min
    true_flux_min = abs(true_flux_min)

    arrow_legnths = true_flux_max - true_flux_min

    height = max(true_flux) + 0.7
    # Plotting Flux Error Graph
    plt.subplot(1, 3, 2)
    plt.bar(x+0.15, val[0], width, color='red', edgecolor ='k', label = 'μ_obs/μ_ref') 
    plt.bar(x-0.15, df_pred[3], width, color='white', edgecolor='k', hatch='\\/', label='μ_pred/μ_ref') 
    plt.bar(x+0.45, true_flux, width, color='salmon', edgecolor='k', label = 'µ_pred/μ_ref (obs pos)')
    # plt.errorbar(x-0.15, val[0], yerr=3*(val[1]), fmt='o', color='black', capsize=4, label='3 σ Error') (OLD ERROR BAR)
    plt.errorbar(x+0.15, val[0], yerr=0.01, fmt='o', color='black', capsize=4, label='1 σ Error')
    plt.xticks(x+0.15, data_df['Label'], fontsize=8) 
    plt.arrow(x[0]+0.45, true_flux[0], 0, arrow_legnths[0], head_width=0.1, head_length=0.05, fc='k', ec='k')
    plt.arrow(x[0]+0.45, true_flux[0], 0, -arrow_legnths[0], head_width=0.1, head_length=0.05, fc='k', ec='k')
    plt.arrow(x[1]+0.45, true_flux[1], 0, arrow_legnths[1], head_width=0.1, head_length=0.05, fc='k', ec='k')
    plt.arrow(x[1]+0.45, true_flux[1], 0, -arrow_legnths[1], head_width=0.1, head_length=0.05, fc='k', ec='k')
    plt.arrow(x[2]+0.45, true_flux[2], 0, arrow_legnths[2], head_width=0.1, head_length=0.05, fc='k', ec='k')
    plt.arrow(x[2]+0.45, true_flux[2], 0, -arrow_legnths[2], head_width=0.1, head_length=0.05, fc='k', ec='k')
    plt.arrow(x[3]+0.45, true_flux[3], 0, arrow_legnths[3], head_width=0.1, head_length=0.05, fc='k', ec='k')
    plt.arrow(x[3]+0.45, true_flux[3], 0, -arrow_legnths[3], head_width=0.1, head_length=0.05, fc='k', ec='k')
    plt.xlabel("Image") 
    plt.ylabel("Flux Ratio") 
    plt.ylim(0, height)
    plt.legend(loc = 'upper right', fontsize='small')
    plt.title('Flux Ratio Error')


    return data_df, df_pred



# Critical Curves Plot

def critcurve_plot(filename_1, filename_2, filename_3, plot_name, num_images):
    data_crit = pd.read_csv(filename_3, header= None, sep="\s+")
    data_crit.__dataframe__
    df = data_crit.iloc[1:]

    # Initialize empty list 
    data = []
    
    # Line by line checking
    with open(filename_1, 'r') as file:
        for line in file:
            # Skip lines starting with #
            if line.startswith("#"):
                continue
            
            # Split by space
            line_data = line.split()
            
            # Remove #
            line_data = [float(val) for val in line_data if val != '#']  # Exclude '#' characters
            
            data.append(line_data)
    
    # Convert the list of lists to a DataFrame
    data_df = pd.DataFrame(data)

    # Exclude the first row
    data_df = data_df.iloc[1:]

    # Read and process the predicted data
    de = pd.read_csv(filename_2, header=None, delim_whitespace=True, comment='#')
    de = de.iloc[1:]

    # Function for swapping data 
    def swap_rows(df, row1, row2):
        df.iloc[row1], df.iloc[row2] =  df.iloc[row2].copy(), df.iloc[row1].copy()
        return df
    
    # For loop to iterate over row range for row swapping
    for i in range(4):
        diff = abs(abs(de.iloc[i,0]) - abs(de[0]))
        m = diff.idxmin()
        n = min(diff)
        if n < 0.01:
            de = swap_rows(de, i, (m-1))
        else:
            continue
        
    de = de.drop(columns =[3])

    # Eliminating the 5th image
    if len(de[2])>num_images:
        i = len(de[2]) - num_images
        for j in range(i):
            min_vales = np.min(abs(de[2]))
            df_4 = abs(de)
            b = df_4.index.get_loc(df_4[df_4[2] == min_vales].index[0])
            df_5 = de.drop((b+1), axis='index')
            de = df_5

    labels = ['A', 'B', 'C', 'D']

    # Plotting Critial Curves
    plt.subplot(1, 3, 3)
    plt.scatter(df[0]*100, df[1]*100, s=1, color = 'orange')
    plt.scatter(df[2]*100, df[3]*100, s=1)
    plt.scatter(df[4]*100, df[5]*100, s=1)
    plt.scatter(df[6]*100, df[7]*100, s=1)

    height_1 = max(df[0]*100) + 80
    height_2 = max(df[1]*100) + 80

    colors = ['red',  'green', 'blue', 'gold']
    # Plotting obs image positions and labels 
    plt.scatter(de[0]*100, de[1]*100, s = 180, marker= '+', label = 'Predicted Position', color = 'black', alpha = 0.5)
    plt.scatter(data_df[0]*100, data_df[1]*100, s=20, color = colors, marker = 'o')
    # for x, y, txt in zip(data_df[0]*100, data_df[1]*100, labels):
    #     plt.text(x, y-17, txt, fontsize=13, ha='center', va='bottom')

    plt.title('Critical Curves')
    plt.legend(loc='upper right', fontsize='small')
    plt.tick_params(labelsize=8)
    plt.xlabel('x [pixel]')
    plt.ylabel('y [pixel]', labelpad=-5)
    plt.xlim(-height_1, height_1)
    plt.ylim(-height_2, height_2)
    plt.suptitle('Lens: ' + plot_name + ' Constrained')
    plt.show()
    
    return data_df, df
