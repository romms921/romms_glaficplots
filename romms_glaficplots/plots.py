import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Position and Magnification Plots

def error_plot(filename, filename_1):
    # Storage for parsed data
    data = []
    
    # Line by line read (Remove # from obs file)
    with open(filename, 'r') as file:
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

    data_df.insert(8, "Label", ['A','B','C','D'], True)

    data_df = data_df.drop(columns =[3, 5, 6, 7])

    # Read and process the predicted data
    data_pred = pd.read_csv(filename_1, header=None, delim_whitespace=True, comment='#')
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

    # Calculations for Position Error values
    d_x = data_df[0]-df_pred[0]
    d_y = data_df[1]-df_pred[1]
    sum_sq = (d_x**2) + (d_y**2)
    sq = np.sqrt(sum_sq)
    rms = np.average(sq)

    # Plotting Position Error Graph
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    colours1 = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    colours = [(0.1,0.4,1.0,0.7), (0.0,0.7,0.4,0.7), (0.7,0.3,0.4,0.7), (0.8,0.7,0.1,0.7)]
    plt.figure(figsize =(8,6))
    plt.bar(data_df['Label'], sq, label=data_df['Label'], color = colours1)
    plt.axhline(xmin=0, xmax=10, y=rms, linestyle ='--', color ='k', linewidth = 2)
    plt.title('SIE Pos Constraint')
    plt.legend(labels=['_', '_', '_', '_', '1 σ Error'])
    plt.xlabel('Image')
    plt.ylabel('Error')
    plt.show()

    # Calculations for Magnification value
    f = np.max(df_pred[2])
    flux = df_pred[2]/f
    df_pred[3] = flux 

    df_pred[3] = abs(df_pred[3])

    # Create data 
    x = np.arange(4)
    width = 0.3

    # Plotting Flux Error Graph
    plt.figure(figsize =(8,6))
    plt.bar(x-0.15, data_df[2], width, color='cyan', edgecolor ='k') 
    plt.bar(x+0.15, df_pred[3], width, color='red', edgecolor='k') 
    plt.errorbar(x-0.15, data_df[2], yerr=data_df[4], fmt='o', color='black', capsize=4, label='1 σ Error')
    plt.errorbar(x-0.15, data_df[2], yerr=2*np.array(data_df[4]), fmt='o', color='black', capsize=4, label='2 σ Error')
    plt.xticks(x, data_df['Label']) 
    plt.xlabel("Image") 
    plt.ylabel("Flux Ratio") 
    plt.legend(labels=['Observed Flux Ratio', 'Predicted Flux Ratio', '1 σ Error', '2 σ Error'])
    plt.title('SIE Pos Constraint Flux Ratio')
    plt.show() 


    return data_df, df_pred



# Critical Curves Plot

def critcurve_plot(filename_4, filename_3):
    data_crit = pd.read_csv(filename_3, header= None, sep="\s+")
    data_crit.__dataframe__
    df = data_crit.iloc[1:]

    # Initialize empty list 
    data = []
    
    # Line by line checking
    with open(filename_4, 'r') as file:
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

    labels = ['A', 'B', 'C', 'D']

    # Plotting Critial Curves
    plt.figure(figsize=(8, 8))
    plt.scatter(df[0]*100, df[1]*100, s=2)
    plt.scatter(df[2]*100, df[3]*100, s=2)
    plt.scatter(df[4]*100, df[5]*100, s=2)
    plt.scatter(df[6]*100, df[7]*100, s=2)

    # Plotting obs image positions and labels 
    plt.scatter(data_df[0]*100, data_df[1]*100, s=20)
    for x, y, txt in zip(data_df[0]*100, data_df[1]*100, labels):
        plt.text(x, y-17, txt, fontsize=13, ha='center', va='bottom')

    plt.title('SIE Position Constraint Critical Curves')
    plt.xlabel('x [pixel]')
    plt.ylabel('y [pixel]')
    plt.xlim(-170,170)
    plt.ylim(-170,170)
    plt.show()
    
    return data_df, df
