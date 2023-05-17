import pickle

with open("vmr_plots.pkl", "rb") as f:
    all_data = pickle.load(f)
    f.close()

epsilons = all_data["eps_list"]
qdist_vmr = all_data["qdist_w_vmr"]
qdist_wo_vmr = all_data["qdist_wo_vmr"]
viol_vmr = all_data["viol_w_vmr"]
viol_wo_vmr = all_data["viol_wo_vmr"]

print(epsilons)
print(qdist_wo_vmr, qdist_vmr)
print(viol_wo_vmr, viol_vmr)
