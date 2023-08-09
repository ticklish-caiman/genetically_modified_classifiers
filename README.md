# genetically_modified_classifiers
This is a simple flask-based app that allows you to experiment with evolutionary algorithms to generate best possible Pipelines for data classification. You can choose between TPOT (well known tree-based tool) and GMC (messy genetic algorithm).
Customize parameters to get best results for chosen dataset. You can include your own data - for now only CSV files are supported.

# Preview
https://user-images.githubusercontent.com/91501936/143504595-230f2cc4-cf10-43dc-abf9-78fb95c1b103.mp4






Tested on Windows 10 only. Some functions are 'hacky' and might not work on other systems.
If you don't care about GUI and want the best possible performance - directly call gmc.evolve() function.
