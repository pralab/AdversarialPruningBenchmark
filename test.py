from utils.utils import load_model, model_key_maker
from utils.plots import plot_sec_curve
from utils.test import test_model_aa, test_model_hofmn
from taxonomy.utils import load_ap_taxonomy

ap = "HARP_Zhao2023Holistic"
arch = "resnet18"
ds = "CIFAR10"
struct = "weights"  # or filters, channels
sr = "90"

# get a unique model key
model_key = model_key_maker(ap_method=ap,
                            architecture=arch,
                            dataset=ds,
                            structure=struct,
                            sparsity_rate=sr)

# when get_distances, the distances computed with the best config from HO-FMN is returned in addition to the model
model = load_model(model_key=model_key)

# test the model
rob_acc_aa = test_model_aa(model)
rob_acc_hofmn, distances = test_model_hofmn(model, loss='DLR', optimizer='SGD', scheduler='CALR', get_distances=True)

# plot security curve
plot_sec_curve(distances, title=model_key+'DLR_SGD_CALR', save=False)

print(f'Model {model_key} AA robust accuracy: {rob_acc_aa}')
print(f'Model {model_key} HO-FMN robust accuracy: {rob_acc_hofmn}')
print(f'Within the taxonomy, here are the AP entries: {load_ap_taxonomy(ap)}')