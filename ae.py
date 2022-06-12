import numpy as np 
import torch
import torch.nn as nn

from topologylayer.nn import AlphaLayer
from gudhi.wasserstein import wasserstein_distance


lifetime = np.vectorize(lambda x: x[1] - x[0], signature='(n)->()')


def k_farthest_points(dgm, k):
    return dgm[np.argsort(lifetime(dgm.detach()))[-k:]]


# https://github.com/aywagner/DIPOLE/blob/main/pers.py
class WassersteinDistance(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p

    def forward(self, dgm1, dgm2):
        print('WassersteinDistance')
        # Diagram is empty => compute total persistance
        if dgm1.shape[0] == 0:
            return torch.sum(torch.pow(dgm2[:, 1] - dgm2[:, 0], self.p))
        if dgm2.shape[0] == 0:
            return torch.sum(torch.pow(dgm1[:, 1] - dgm1[:, 0], self.p))

        dgm1_np, dgm2_np = dgm1.detach().numpy(), dgm2.detach().numpy()
        # Compute optimal matching using GUDHI
        matching = wasserstein_distance(dgm1_np, dgm2_np, matching=True, order=self.p, internal_p=1)[1]
        # Initialize cost
        cost = torch.tensor(0., requires_grad=True)
        # Note these calculations are using L1 ground metric on upper half-plane
        is_unpaired_1 = (matching[:, 1] == -1)
        if np.any(is_unpaired_1):
            unpaired_1_idx = matching[is_unpaired_1, 0]
            cost = cost + torch.sum(torch.pow(dgm1[unpaired_1_idx, 1] - dgm1[unpaired_1_idx, 0], self.p))
        is_unpaired_2 = (matching[:, 0] == -1)
        if np.any(is_unpaired_2):
            unpaired_2_idx = matching[is_unpaired_2, 1]
            cost = cost + torch.sum(torch.pow(dgm2[unpaired_2_idx, 1] - dgm2[unpaired_2_idx, 0], self.p))
        is_paired = (~is_unpaired_1 & ~is_unpaired_2)
        if np.any(is_paired):
            paired_1_idx, paired_2_idx = matching[is_paired, 0], matching[is_paired, 1]
            paired_dists = torch.sum(torch.abs(dgm1[paired_1_idx, :] - dgm2[paired_2_idx, :]), dim=1)
            paired_costs = torch.sum(torch.pow(paired_dists, self.p))
            cost = cost + paired_costs
        return torch.pow(cost, 1. / self.p)


class DistTopoLoss(nn.Module):
    def __init__(self, data, hom_weights, subset_size, k, p):
        """
        data: target point cloud
        hom_weights: tuple of weights for persistent homology degrees
        subset_size: size of randomly sampled subsets
        k: number of significant points on diagrams
        p: choice of Wasserstein distance raised to p
        """
        super().__init__()
        self.data = data
        self.alpha_layer = AlphaLayer(maxdim=2)
        self.hom_weights = hom_weights
        self.subset_size = subset_size
        self.k = k
        self.wasserstein = WassersteinDistance(p)

    def forward(self, emb):
        print('DistTopoLoss')
        indices = np.random.randint(self.data.shape[0], size=self.subset_size)
        subset, emb = self.data[indices, :], emb[indices, :]
        subset_dgms, _ = self.alpha_layer(subset)
        emb_dgms, _ = self.alpha_layer(emb)
        topo_loss = torch.tensor(0., requires_grad=True)
        for i in [0, 1, 2]:
            topo_loss = topo_loss + self.hom_weights[i] * \
                                    self.wasserstein(k_farthest_points(subset_dgms[i][max(1 - i, 0):], self.k),
                                                     k_farthest_points(emb_dgms[i][max(1 - i, 0):], self.k))
        print('OK')
        return topo_loss


class LocalMetricRegularizer(nn.Module):
    def __init__(self, dist_mat, mask):
        super().__init__()
        self.indices = torch.nonzero(mask)
        self.small_dists = dist_mat[mask]
         
    def forward(self, emb):
        print('LocalMetricRegularizer')
        emb_diffs = emb[self.indices[:, 0]] - emb[self.indices[:, 1]]
        emb_small_dists = torch.linalg.norm(emb_diffs, dim=1)
        return ((self.small_dists - emb_small_dists) ** 2).sum()


class TotalPersistenceLoss(nn.Module):
    def __init__(self, data, k, m): # hom_weights?
        super().__init__()
        self.k = k
        self.m = m
        self.alpha_layer = AlphaLayer(maxdim=2)
        self.wasserstein = WassersteinDistance(m)
        data_dgms, _ = self.alpha_layer(data)
        self.data_tot_pers = torch.zeros(3, dtype=torch.float)
        for i in [0, 1, 2]:
            # Total persistence is a special case of Wasserstein distance
            self.data_tot_pers[i] = self.wasserstein(k_farthest_points(data_dgms[i][max(1 - i, 0):], self.k), np.array([]))

    def forward(self, emb):
        print('TotalPersistenceLoss')
        emb_dgms, _ = self.alpha_layer(emb)
        tot_pers_loss = torch.tensor(0., requires_grad=True)
        for i in [0, 1, 2]:
            tot_pers_loss = tot_pers_loss + \
                            (self.wasserstein(
                                k_farthest_points(emb_dgms[i][max(1 - i, 0):], self.k),
                                np.array([])
                            ) - self.data_tot_pers[i]) ** 2
        return tot_pers_loss


class LinearTopologyPreservingAE(nn.Module):
    def __init__(self, data, input_dim, latent_dim, topo_coef, lmr_coef, tot_pers_coef,
                 n_subsets, subset_size, hom_weights, k, p, m, delta, seed):
        super().__init__()
        # self.input_dim = input_dim
        # self.latent_dim = latent_dim
        self.topo_coef = topo_coef
        self.lmr_coef = lmr_coef
        self.tot_pers_coef = tot_pers_coef
        self.n_subsets = n_subsets
        # self.hom_weights = hom_weights
        # self.k = k
        # self.m = m
        # self.delta = delta

        torch.manual_seed(seed)
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

        dist_mat = torch.cdist(data, data)
        mask = (0 < dist_mat) & (dist_mat < delta)

        self.reconst_loss = nn.MSELoss()
        self.topo_loss = DistTopoLoss(data, hom_weights, subset_size, k, p)
        self.lmr = LocalMetricRegularizer(dist_mat, mask)
        self.tot_pers_loss = TotalPersistenceLoss(data, k, m)

    def encode(self, X):
        return self.encoder(X)

    def decode(self, Z):
        return self.decoder(Z)

    def forward(self, X):
        Z = self.encode(X)
        X_reconst = self.decode(Z)

        reconst_loss = self.reconst_loss(X, X_reconst)
        topo_loss = torch.tensor(0., requires_grad=True)
        for _ in range(self.n_subsets):
            print(_)
            topo_loss = topo_loss + self.topo_loss(Z)
        lmr_loss = self.lmr(Z)
        tot_pers_loss = self.tot_pers_loss(Z)

        return reconst_loss + self.topo_coef * \
               (topo_loss / self.n_subsets + \
               self.lmr_coef * lmr_loss + \
               self.tot_pers_coef * tot_pers_loss)
