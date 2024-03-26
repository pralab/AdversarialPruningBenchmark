# Adversarial Pruning Benchmark :grapes: :scissors: :shield:

**Giorgio Piras (University of Cagliari, University of Roma La Sapienza), Maura Pintor (University of Cagliari), Ambra Demontis (University of Cagliari), Battista Biggio (University of Cagliari), Giorgio Giacinto (University of Cagliari), Fabio Roli (University of Cagliari, University of Genova)**

**Paper:** [link](link)

## Main idea 
- The goal of the Adversarial Pruning (AP) Benchmark is to make AP methods be evaluated in a fair, comparable way :monocle_face: 

- APs represent all those pruning methods whose goal is to get a pruned network while preserving/inducing as much robustness against adversarial attacks as possible :shield:

- That is why, in our [paper](paper), we created a taxonomy of adversarial pruning method, thus enabling a clear and systematic analysis of the methods :bookmark_tabs:

- In addition, to fairly and comparably analyze the AP methods, we created the adversarial pruning benchmark :hammer_and_wrench: :grapes:

#### We evaluate the AP methods using the AutoAttack framework and the HO-FMN attack, making up for the major issues in adversarial evaluations and ensuring a complete security evaluation :star_struck: 




## Usage: Do you want to taste the grapes? :grapes: 
### Here's how you test a pruned model:
:video_game: To test a pruned model that is available at our leaderboard, one must specify the AP method, the architecture, the dataset, the structure, and the sparsity.
Then, the model can be loaded and tested, and additionally security curves can be plotted! 

```python
from models import *
from attacks import * 
from utils import load_model, test_model
from utils.plots import plot_sec_curve
from taxonomy import load_taxonomy

ap = "HARP_Zhao2023Holistic"
arch = "resnet18" 
ds = "CIFAR10" 
struct= "weights" # or filters, channels
sr = "90"

# when get_distances, the distances computed with the best config from HO-FMN is returned in addition to the model
model, distances = load_model(ap_method = ap, 
                              architecture = arch, 
                              dataset = ds, 
                              structured = struct, 
                              sparsity_rate = sr, 
                              get_distances = True) 

rob_acc_aa, rob_acc_hofmn = test_model(model, aa=True, hofmn=True) 
plot_sec_curve(distances, title='HARP_R18_CIFAR10_US_90', save=False)

print(f'The AP {ap} has AutoAttack robust accuracy equal to {rob_acc_aa}') 
print(f'Within the taxonomy, here are the AP entries: {load_taxonomy(ap)}') 

```



## Contributing: Did you just harvest? :scissors: 
### Here's how to load an AP in our benchmark:
:hugs: We welcome AP authors wishing to see their AP be taxonomized and benchmarked! 
To load AP methods in our benchmark, one must compile the taxonomy in our [Google Drive form](), and then submit the checkpoint files at... :warning: :construction_worker_man:
Then, the checkpoints must be evaluated using the `test_model()` method. 


## Install our repository

First, install the latest version of **`RobustBench`** (recommended):

```bash
git clone "link/to/repo@github.com"
```


### CIFAR-10 US pruning, ResNet18

#### Linf, eps=8/255
| <sub>#</sub> | <sub>ap ID</sub>                                      | <sub>Paper</sub>                                                                                              | <sub>Clean accuracy</sub> | <sub>Robust accuracy</sub> | <sub>Sparsity</sub> |   <sub>Venue</sub>   |
|:---:|-------------------------------------------------------|---------------------------------------------------------------------------------------------------------------|:-------------------------:|:--------------------------:|:-------------------:|:--------------------:|
| <sub>**2**</sub> | <sub><sup>**FlyingBird_Chen2021Sparsity**</sup></sub> | <sub>*[Sparsity Winning Twice: Better Robust Generalization from More Efficient Training](https://openreview.net/pdf?id=SYuJXrXq8tw)*</sub> |     <sub>80.69</sub>      |     <sub>46.69%</sub>      |    <sub>90</sub>    | <sub>ICLR 2022</sub> |
| <sub>**1**</sub> | <sub><sup>**HARP_Zhao2023Holistic**</sup></sub>       | <sub>*[Holistic Adversarially Robust Pruning](https://intellisec.de/pubs/2023-iclr.pdf)*</sub>                        |     <sub>83.38%</sub>     |     <sub>45.40%</sub>      |    <sub>90</sub>    | <sub>ICLR 2023</sub> |

### CIFAR-10 US pruning, VGG16
#### Linf, eps=8/255

### CIFAR-10 S pruning, ResNet18
#### Linf, eps=8/255

### CIFAR-10 S pruning, VGG16
#### Linf, eps=8/255

### SVHN US pruning, ResNet18
#### Linf, eps=8/255

### SVHN US pruning, VGG16
#### Linf, eps=8/255

### SVHN S pruning, ResNet18
#### Linf, eps=8/255

### SVHN S pruning, VGG16
#### Linf, eps=8/255

## How to contribute
We welcome any contribution to the Benchmark, such as:

- Adding novel AP methods, with their checkpoints.
- Discussing the taxonomy pillars. 
- Discussing the benchmark experimental setup. 
- Discussing new and different adversarial attacks for evaluating the pruned models.


## Adding a new AP method

## Adding a new model

## Citation


```bibtex

}
```

## Contact
Feel free to contact us about anything related to our benchmark by creating an issue, a pull request or
by email at `giorgio.piras@unica.it`.
