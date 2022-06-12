import numpy as np 
import torch
from ae import LinearTopologyPreservingAE, lifetime


def train_ae(X, target_dim, topo_coef=1, lmr_coef=0.001, tot_pers_coef=0.1,
             # n_subsets=2, subset_size=32, hom_weights=(0.4, 0.4, 0.2), k=5, p=2, m=1, delta=50,
             n_iterations=100, lr=0.01, seed=852, plot_dgms=False, save_dgms=False):
    model = LinearTopologyPreservingAE(
        data=X, input_dim=X.shape[1], latent_dim=target_dim,
        topo_coef=topo_coef, lmr_coef=lmr_coef, tot_pers_coef=tot_pers_coef,
        n_subsets=2, subset_size=64, hom_weights=(0.4, 0.4, 0.2),
        k=5, p=2, m=1, delta=50
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = ...
    loss_history = []
    np.random.seed(seed)
    for i in range(n_iterations):
        optimizer.zero_grad()
        loss = model(X)
        loss_history.append(loss.detach().item())
        loss.backward()
        optimizer.step()

        # if not plot_dgms:
        #     continue
        # if i == 1 or i % 10 == 0:
        #     with torch.no_grad():
        #         emb = model.encode(X)
        #         emb_dgms, _ = ...
        #         plt.figure(figsize=(3.5, 3.5))
        #         k_far_dmgY = dgmY[1][np.argsort(lifetime(dgmY[1].detach()))[-5:]].detach()
        #         plot_diagrams([k_far_dgmX.numpy(), k_far_dmgY.numpy()], labels=['$H_1(X)$', '$H_1(Y)$'])
        #         if save_dgms:
        #             plt.savefig(f'diags_iter_{i}.pdf', format='pdf')
        #             plt.clf()
    return loss_history
