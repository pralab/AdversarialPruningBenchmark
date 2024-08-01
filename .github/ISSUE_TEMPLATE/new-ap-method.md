---
name: New AP method
about: This issue allows adding a new AP method
title: ''
labels: ''
assignees: ''

---

# General Information 
As we describe in detail in the README, diversity is fundamental. We thus require AP authors to load a minimum of 3 checkpoints, one for sparsity for a single arch/dataset/structure pair. 
If you wish to load only 3 checkpoints, you are however strongly encouraged to take as a first step, and load the remaining (ideally, 9) models at your best convenience. 
In this template, you are required to have the information sent to your address after having compiled the taxonomy (the JSON Entry), as well as the results given by the benchmark evaluation.  

# Loading Section 

## JSON Entry 
Put here the raw JSON Entry received by email. 

## Evaluation Results of Model n°1
Please, fill each model's section as follows:
"model_key_from_benchmark" : ["your_gdrive_link_id", "extension"] 
Clean Accuracy: xx.xx 
Robust Accuracy (AA): xx.xx 
Robust Accuracy (HO-FMN): xx.xx 

Please note: The gdrive link is of your property, and you will therefore need to make the download available for everyone. Also, only the id of the link is required (e.g., `["1uDif9I7iROSAil8gZG0tJGbHiBrvK--h", ".pt"]`). 

## Evaluation Results of Model n°2
Same procedure of model n°1

## Evaluation Results of Model n°3 
Same procedure of model n°1

## Architecture details and normalization 
Please, write here if, and which of your loaded models need normalization. 
Also, write if your checkpoints cannot be loaded into the base models available. This is common for pruning methods, as the masks often require creating specific layers and we do not always feel like multiplying the mask and save the dense models with standard layers :). 
It is, however, good practice to have the same