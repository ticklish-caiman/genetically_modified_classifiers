# genetically_modified_classifiers
<p align="justify">
This is a simple flask-based app that allows you to experiment with evolutionary algorithms to generate best possible Pipelines for data classification. You can choose between TPOT (well known tree-based tool) and GMC (messy genetic algorithm).
Customize parameters to get best results for chosen dataset. You can include your own data - for now only CSV files are supported.
</p>

# GUI Preview
https://user-images.githubusercontent.com/91501936/143504595-230f2cc4-cf10-43dc-abf9-78fb95c1b103.mp4


# Sample experiments results 
<p align="justify">
Randomness is an underlying property of genetic algorithms. 
Each run, a random population is generated (different type of classifiers with various parameters). Each classifier can have different computational complexity – some will work great for datasets with many objects, but less attributes; others the other way around. 
But the general tendency is simple:
</p>

    • small dataset and small populations = fast results
    • large dataset and big populations = slow results

<p align="justify">
TPOT allows you to control initial population size and the offspring size. In GMC you can achieve similar effect using elitism. Below you can see how different options may effect the running time of the algorithm (first value is the initial population size/second is the offspring size).
</p>

![qsar_times](https://github.com/ticklish-caiman/genetically_modified_classifiers/assets/91501936/2b8610c5-f1ae-4589-89fa-13d1e5bb74a5)

Note that in that case we are using an early stop condition: no improvement in n-generations stops the evolution process.

Alright – training more than a hundred classifiers each generations will take a lot of time, that's for sure… but is it worth it? 

![qsar_accu](https://github.com/ticklish-caiman/genetically_modified_classifiers/assets/91501936/88ad3864-ce29-4332-b730-ba166577a198)
<p align="justify">
Each population/offspring combination above has been run 10 times. The best classifier was obtained with 64/64 combination. Even on average that combination was better than much slower 128/128. However, 10 runs is not enough to draw final conclusions: we are dealing with randomness on many levels. As usual: further research is needed.

So let’s do the same with few other datasets, aggregate the results to a common denominator and plot everything on a single graph:
</p>

![populations_normilized](https://github.com/ticklish-caiman/genetically_modified_classifiers/assets/91501936/09c12fc8-ec53-448d-9ff3-e223b2e472e0)

  
![obraz](https://github.com/ticklish-caiman/genetically_modified_classifiers/assets/91501936/ce8273cc-a7d9-4622-9816-c2d6fa47e58f)
<p align="justify">That function simply makes the best result a 1 and measures how much worse the other results are.
It would seem that for used datasets option 128/128 is not much better than 32/32.
Note that for each experiment we are using cross-validation x10 – that means each time we divide the whole set 10 times and train on each subset. 
Let’s say that one CV takes (on average) 1 second – seems fast, right? So for the 128/128 and 1000 generations we have to conduct 128128 cross-validations… it will take approximately 35 hours… times 10… it would be less than 15 days. 
All my experiments took a while, but thanks to early-stop conditions it wasn’t that bad.
</p>

# Important question

Is GMC better than TPOT?

  Short answer: no. 

<p align="justify">
The amount of possible parameters and the nature of genetic algorithm, as well as the task itself (different datasets) makes testing very difficult. 
GMC was made and tested by one person within a two year period. So far most of the tests indicate that you can achieve better results with TPOT. 
I only hope that some day, someone will have fun with GMC and maybe appreciates its messiness.
</p>

# More testing is needed 
<p align="justify">
Tested on Windows 10 only. Some functions are 'hacky' and might not work on other systems.
If you don't care about GUI and want the best possible performance - directly call gmc.evolve() function.
</p>
