# Creative problem solving in LLMs
This repository contains data and code for _preliminary_ experiments demonstrating creative problem solving in LLMs, inspired by Computational Creativity literature. The code provided in this repository evaluates the capabilities of LLMs to identify creative object replacements when the required objects are missing, e.g., substituting a bowl for a scoop. The approach evaluates performances of the LLMs for different prompts such prompts that are augmented with relevant object features.

## Instructions for running the code
To run the code:
```
python creative-problem-solving/eval_task.py --task-type creative-obj
```
Details of the models and task prompts are available in `dataset_cfg.py`. The task types include `creative-obj` that adds object feature information to the prompt; `creative-task` that adds task information to the prompt; and `creative-task-obj` that combines object and task information. Additionally `nominal` uses regular prompts, tested on cases where object replacement is not required, and `creative` tests the models with regular prompts in cases where an object replacement is required. The code runs the evaluation (creating random test sets based on a seed) and reports the result via plots as shown below. If the code is run as is with the default seed setting, the following plots will be generated for each task type. 

### Using the `creative` prompt
![alt text](assets/Viz_creative.png "")

### Using the `creative-obj` prompt
![alt text](assets/Viz_creative-obj.png "")

### Using the `creative-task` prompt
![alt text](assets/Viz_creative-task.png "")

### Using the `creative-task-obj` prompt
![alt text](assets/Viz_creative-task-obj.png "")

The full testing dataset consists of 16 RGB images of objects, from which subsets are randomly chosen.
![alt text](assets/artificial-dataset.png "")

# References
If you find this repository useful, kindly consider citing the following paper:
```
[TBD]
```

# Acknowledgements
This work is inspired by the Object-Replacement-Object-Composition framework from Computational Creativity - see `Olteteanu and Falomir, 2016: Object replacement and object composition in a creative cognitive system. Towards a computational solver of the alternative uses test`
