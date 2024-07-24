
# Glafic Graph Plotter

This is a simple Python package that automates making the basic position/flux error and critical curve plots. 


## Authors

- [@rommuluslewis](https://github.com/romms921)
- [@shashpalsingh](https://github.com/Shashpal)


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Installation

Install romms_glaficplots with pip

```bash
  pip install romms_glaficplots
```
    
## Usage/Examples

```python
from romms_glaficplots import error_plot, critcurve_plot

filename_0 = 'test_point.py'
filename_1 = 'obs_point.dat'
filename_2 = 'SIE_POS_point.dat'
filename_3 = 'SIE_POS_crit.dat'
filename_4 = 'flux.dat'
filename_5 = 'SIE_POS_lens.fits'
plot_name = 'SIE(POS)'
num_images = 4

obs_data, pred_data = error_plot(filename_1, filename_2, filename_4, filename_5, plot_name, num_images, table_flag=True, glafic_file= filename_0)

obs_data, pred_data = critcurve_plot(filename_1, filename_2, filename_3, plot_name, num_images)
```

## Package Function Use
This package can be used to generate only plots or the parameter table or both. Use the table_flag function to generate the table along with the plot and specify the glafic_file (only works for glafic running in python). If you only require the plots you do not need to specify table_flag or galfic_file and can ignore both. If you require only the table use the glafic_tabular function for the same. If you would like to save the table as a csv use the save_table_flag.

```glafic_tabular(filename0, save_table_flag = True)```

## Flux File Format Example
This file specifies the observed flux ratios. (Values specified according to Neirenberg).

example_flux.dat has an example of the format for the flux file. 

## Acknowledgements 
Special thanks to Shashpal Singh for the format and styling of the plots.
