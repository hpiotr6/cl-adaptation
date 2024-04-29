from matplotlib import pyplot as plt
import numpy as np
import torch


def calculate_eigens_directclr(embeddings):
    embeddings = embeddings.cpu().detach().numpy()
    embeddings = np.transpose(embeddings)
    cov = np.cov(embeddings)

    _, d, _ = np.linalg.svd(cov)
    return d


def torch_calculate_eigens_directclr(embeddings):
    embeddings = embeddings.T

    cov = torch.cov(embeddings)

    d = torch.linalg.svd(cov)
    return d


# def plot_eigenspectrum_v2(dataset: str, ckpts: dict):
#     columns = len(ckpts)

#     plt.rcParams.update({"font.size": 16})

#     fig, axs = plt.subplots(1, columns, dpi=100, figsize=(columns * 5, 1 * 5))

#     train_loader, val_loader = prepare_data(
#         dataset,
#         data_dir="data",
#         train_dir=None,
#         val_dir=None,
#         batch_size=1000,
#         num_workers=4,
#         semi_supervised=None,
#     )

#     for col, (exp_name, _ckpts) in enumerate(ckpts.items()):
#         for task_idx, ckpt in enumerate(_ckpts):
#             backbone = get_initialized_backbone(ckpt)
#             embeddings = get_embeddings(backbone, val_loader)
#             # eigen_value, _ = calculate_eigens(embeddings)
#             # df = pd.DataFrame()
#             # eigs = eigen_value.cpu()
#             # df['eigs'] = eigs.log().numpy()
#             # df['xs'] = list(range(len(eigs)))

#             eigs = calculate_eigens_directclr(embeddings)
#             df = pd.DataFrame()
#             df["eigs"] = np.log(eigs / eigs[0])
#             df["xs"] = list(range(len(eigs)))

#             ax = axs[col]

#             ax.plot(df["xs"].values, df["eigs"].values, label=f"Task {task_idx+1}")

#         ax.legend(loc="upper right", fancybox=True, shadow=False)
#         # ax.set_title(exp_name + f" on dataset {dataset}")
#         ax.set_ylim([-10, 0.5])
#         ax.set_xlabel("Singular Value Rank Index")
#         if col == 0:
#             ax.set_ylabel("Log of normalized singular values")
#         if col != 0:
#             ax.set(ylabel="", yticklabels=[])

#     fig.tight_layout()
