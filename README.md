# genetically_modified_classifiers
This is a simple flask-based app that allows you to experiment with evolutonary algorithms to generate the best possible Pipelines for data classification. You can choose between TPOT (well known tree-based tool) or GMC (messy genetic algorithm).
Customize parameters to get best results for chosen dataset. You can include your own data - for now only CSV files are supported.

# Preview
https://user-images.githubusercontent.com/91501936/135160452-1f2413b0-ed31-48e4-a529-7b5fe751cc5e.mp4



Tested on Windows 10 only. Some functions are 'hacky' and might not work on other systems.
If you want the best possible performance - skip flask and directly call GMC.evolve() function.
