import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings
warnings.filterwarnings('ignore')

# Glafic Tabular 

# Read_script.py
# Open the Python file as a text file
def glafic_tabular(filename_0, save_table_flag = False):
    with open(filename_0, 'r') as file:
        # Read the contents of the file
        content = file.read()

    # Define a function to find a line in the file
    def find_line(word, content):
        content = content.split('\n')
        line_number = 0
        for line in content:
            line_number += 1
            if word in line:
                return line_number
        return "Line Not Found"

    # Set lens line number
    line_set = find_line('glafic.set_lens', content)
    line_opt = find_line('glafic.setopt_lens', content)

    if line_set == "Line Not Found" or line_opt == "Line Not Found":
        raise ValueError("Failed to find lens or setopt lens lines in the glafic file.")

    # Split the content by new line
    content_list = content.split('\n')

    # Get the set_lens line
    set_lens = content_list[line_set-1]

    # Get the setopt_lens line
    setopt_lens = content_list[line_opt-1]

    # Define the list of possible models
    models = ['SIE', 'POW', 'NFW']

    parts_set_lens = set_lens.split(',')
    parts_set_lens = [part.strip().strip("'") for part in parts_set_lens]

    parts_setopt_lens = setopt_lens.split(',')
    parts_setopt_lens = [part.strip().strip("'") for part in parts_setopt_lens]

    for i in models:
        i = i.lower()
        if i in set_lens:
            if i == models[1].lower(): # POW model
                name = models[1]
                x = parts_set_lens[4]
                y = parts_set_lens[5]
                e = parts_set_lens[6]
                pa = parts_set_lens[7]
                r_ein = parts_set_lens[8]
                pwi = parts_set_lens[9].replace(')', '') 

                x_flag = parts_setopt_lens[3]
                y_flag = parts_setopt_lens[4]
                e_flag = parts_setopt_lens[5]
                pa_flag = parts_setopt_lens[6]
                r_ein_flag = parts_setopt_lens[7]
                pwi_flag = parts_setopt_lens[8].replace(')', '') 

                row_0 = [name, 'x', 'y', 'e', 'θ', 'r_ein', 'γ (PWI)']
                row_1 = ['Value', x, y, e, pa, r_ein, pwi]
                row_2 = ['Fixed', x_flag, y_flag, e_flag, pa_flag, r_ein_flag, pwi_flag]

                table = pd.DataFrame([row_1, row_2], columns = row_0)

            elif i == models[0].lower(): # SIE model
                name = models[0]
                sigma = parts_set_lens[3]
                x = parts_set_lens[4]
                y = parts_set_lens[5]
                e = parts_set_lens[6]
                pa = parts_set_lens[7]
                r_core = parts_set_lens[8]
                pwi = parts_set_lens[9].replace(')', '') 

                sigma_flag = parts_setopt_lens[2]
                x_flag = parts_setopt_lens[3]
                y_flag = parts_setopt_lens[4]
                e_flag = parts_setopt_lens[5]
                pa_flag = parts_setopt_lens[6]
                r_core_flag = parts_setopt_lens[7]
                pwi_flag = parts_setopt_lens[8].replace(')', '') 

                row_0 = [name, 'σ', 'x', 'y', 'e', 'θ', 'r_core', 'γ (PWI)']
                row_1 = ['Value', sigma, x, y, e, pa, r_core, pwi]
                row_2 = ['Fixed', sigma_flag, x_flag, y_flag, e_flag, pa_flag, r_core_flag, pwi_flag]

                table = pd.DataFrame([row_1, row_2], columns = row_0)

            elif i == models[2].lower(): # NFW model    
                name = models[2]
                m = parts_set_lens[3]
                x = parts_set_lens[4]
                y = parts_set_lens[5]
                e = parts_set_lens[6]
                pa = parts_set_lens[7]
                c = parts_set_lens[8].replace(')', '') 

                m_flag = parts_setopt_lens[2]
                x_flag = parts_setopt_lens[3]
                y_flag = parts_setopt_lens[4]
                e_flag = parts_setopt_lens[5]
                pa_flag = parts_setopt_lens[6]
                c_flag = parts_setopt_lens[7].replace(')', '') 

                row_0 = [name, 'M_tot', 'x', 'y', 'e', 'θ_e', 'c']
                row_1 = ['Value', m, x, y, e, pa, c]
                row_2 = ['Fixed', m_flag, x_flag, y_flag, e_flag, pa_flag, c_flag]

                table = pd.DataFrame([row_1, row_2], columns = row_0)

            if save_table_flag:
                table.to_csv('table.csv')
            
            return table  
    
    print("Model not found")
    return None  # Explicitly return None if no model is found
            

# Position and Magnification Plots

def error_plot(filename_1, filename_2, filename_4, filename_5, plot_name, num_images, table_flag = False, glafic_file=None):
    if table_flag:
        if glafic_file is None:
            print("Please provide the filename for the glafic script")
            raise ValueError("Glafic File not provided")

    if table_flag:    
        table = glafic_tabular(glafic_file)
        if table is None:
            raise ValueError("Failed to create the table from the glafic file.")

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
            df_pred.reset_index(drop=True, inplace=True)
        df_pred.index = df_pred.index + 1

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
    plt.suptitle('Lens: ' + plot_name + ' Constrained')
    if table_flag:
        table_plot = plt.table(cellText=table.values, colLabels=table.columns, cellLoc = 'center', loc='bottom', bbox=[-1.0, -0.5, 3.0, 0.3])
        table_plot.auto_set_font_size(False)
        table_plot.set_fontsize(10)

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
            de.reset_index(drop=True, inplace=True)
        de.index = de.index + 1

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