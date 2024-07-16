
# Glafic Graph Plotter

This is a simple python package that automates making the basic position/flux error and ciritcal curve plots. 


## Authors

- [@rommuluslewis](https://github.com/romms921)


## License

[MIT](https://choosealicense.com/licenses/mit/)


## Installation

Install romms_glaficplots with pip

```bash
  pip install romms_glaficplots
```
    
## Usage/Examples

```python
import romms_glaficplots

filename_1 = 'obs_point_SIE(POS).dat'
filename_2 = 'SIE_test_point.dat'
filename_3 = 'SIE_test_crit.dat'
filename_4 = 'flux.dat'
filename_5 = 'SIE_test_lens.fits'
plot_name = 'SIE (POS)'
num_images = 4

obs_data, pred_data = error_plot(filename_1, filename_2, filename_4, filename_5, plot_name, num_images)

obs_data, pred_data = critcurve_plot(filename_1, filename_2, filename_3, plot_name, num_images)
```

