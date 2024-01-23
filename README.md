# Creative problem solving in LLMs
This repository contains data and code for _preliminary_ experiments demonstrating creative problem solving in LLMs, inspired by Computational Creativity literature. The code provided in this repository evaluates the capabilities of LLMs to identify creative object replacements when the required objects are missing, e.g., substituting a bowl for a scoop. The approach evaluates performances of the LLMs for different prompts, e.g., prompts that are augmented with relevant object features.

## Instructions for running the code
This repository requires [Pytorch](https://github.com/pytorch/pytorch) and HuggingFace [Transformers](https://github.com/huggingface/transformers) libraries. 
```
pip install -r requirements.txt
```

Running with the default seed setting (`seed=42`) will reproduce results from the paper, shown in the images below. To run the code:
```
python creative-problem-solving/eval_task.py --task-type creative-obj
```
Details of the models and task prompts are available in `dataset_cfg.py`. The task types include `creative-obj` that adds object feature information to the prompt; `creative-task` that adds task information to the prompt; and `creative-task-obj` that combines object and task information. Additionally `nominal` uses regular prompts, tested on cases where object replacement is not required, and `creative` tests the models with regular prompts in cases where an object replacement is required. The code runs the evaluation (creating random test sets based on the seed) and reports the result via plots as shown below.

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

# Acknowledgements
