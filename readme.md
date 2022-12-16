
# NodalMOD
[NodalMOD](https://github.com/pauleseifert/NodalMOD) is a [Python](https://python.org/) framework for the allocation and disaggregation of electricity system data to the nodal level of a given grid system. The main purpose is to attain a high spatial resolution needed for investigations into flexibility and high-capacity projects with a simple grid representation and interpolated data for future years. 
Scenarios and the objective function can easily be modified.

The branches contain multiple configurations with zonal and nodal dispatch. In the first analysis under the title of *How to Connect Energy Islands: Trade-offs Between Hydrogen and Electricity Infrastructure* (Lüth, Seifert, Egging-Bratseth, Weibezahn 2022), we investigated optimal investments into electricity cable and electrolyser capacity for the integration of the large wind power plants at the energy islands in the north and baltic sea. The code is available under the branch "Energy_Islands". 

Questions, suggestions, or own contributions are highly appreciated! 

## Documentation
*work in progress*
### Using the model
1. To start the workflow, edit the run parameters in the *data_object.py* file. You can define the years and time steps the model should run.
2. Create a .mps model for gurobi by running the *model_description.py* file
3. Solve the model by running the *model_solving.py* file
4. Create maps, plots and Excel files for further investigations and analyses by running the *scen_kpi.py*.
