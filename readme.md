This folder contains supplementary data for the "Sensitivity of dryland vegetation patterns to storm characteristics", under review at GRL

List of folders:

- animations:
 
	Sample pattern animations for the exploratory simulations with R = 282 mm/year

src:  
 	miscellaneous python code used by the jupyter notebook files

notebook_files : 
	jupyter notebook files to visualize: 
	(i) the SVE training simulations (`View_SVE_training_sims.ipynb`)
	(ii) emulator model features (`Random_forest_features.ipynb`)

SVE_tr-10,d-1.6 :
	plots of all the training simulations with storm duration = 40 min, and rain depth = 1.6 cm. 
	All other training simulations can be visualized in 


validation_figures :
	contains side-by-side comparisons of the R/G-SVE and R/G-emulator biomass predictions after 632 storms (from the validation simulations)

List of data files:

SVE_training_data.pkl : dataset of SVE model simulations used to train the emulator models. These simulations can be viewed in the notebook file: `View_SVE_training_sims.ipynb`


