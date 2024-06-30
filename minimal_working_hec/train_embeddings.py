import os
import time
from datetime import datetime

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from geoopt.manifolds import PoincareBallExact
from geoopt.optim import RiemannianSGD

from hyphierarchy.datasets.load_graphs import load_graph_from_file
from hyphierarchy.datasets.dataset import HierarchyEmbeddingDataset

from hyphierarchy.embeddings.poincare_embeddings.embedding import PoincareEmbedding
from hyphierarchy.embeddings.hyperbolic_entailment_cones.embedding import EntailmentConeEmbedding
from hyphierarchy.embeddings.distortion.embedding import DistortionEmbedding

# from hyphierarchy.utils.dist_mat_plots import create_dist_mat_plots, create_dist_diff_mat_plots


if __name__ == "__main__":
    # File paths
    cwd = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(cwd, "data")
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = os.path.join(cwd, "experiments", "embeddings", now)
    os.makedirs(exp_dir)

    # Hyperparameters
    lr = 1e0
    burn_in_lr_mult = 1 / 10
    epochs = 10000
    burn_in_epochs = 20
    embedding_dim = 2


    # Load hierarchy and wrap into dataloader
    hierarchy = load_graph_from_file(os.path.join(data_dir, "cifar100.json"))
    # hierarchy.remove_node(127)
    dataset = HierarchyEmbeddingDataset(
        hierarchy=hierarchy,
        num_negs=10,
        edge_sample_from="both",
        edge_sample_strat="uniform",
        dist_sample_strat="shortest_path",
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=120,
        shuffle=True,
    )

    # Initialize embedding model
    ball = PoincareBallExact(c=1.0)
    # TODO: this number of embeddings is probably wrong. Still has the root node.
    # model = EntailmentConeEmbedding(
    #     num_embeddings=hierarchy.number_of_nodes(),
    #     embedding_dim=embedding_dim,
    #     ball=ball,
    # )
    # model = PoincareEmbedding(
    #     num_embeddings=hierarchy.number_of_nodes(),
    #     embedding_dim=embedding_dim,
    #     ball=ball,
    # )
    model = DistortionEmbedding(
        num_embeddings=hierarchy.number_of_nodes(),
        embedding_dim=embedding_dim,
        ball=ball,
    )

    # Initialize optimizer
    optimizer = RiemannianSGD(
        params=model.parameters(),
        lr=lr,
        momentum=0,
        weight_decay=0,
    )

    # Train the model
    start = time.time()
    losses, _ = model.train(
        dataloader=dataloader,
        epochs=epochs,
        optimizer=optimizer,
        burn_in_epochs=burn_in_epochs,
        burn_in_lr_mult=burn_in_lr_mult,
        store_losses=True,
    )
    print(f"Elapsed training time: {time.time() - start:.3f} seconds")
    print(os.path.join(exp_dir, f"{model.__class__.__name__}_weights_{embedding_dim}.pth"))
    torch.save(
        obj=model.state_dict(),
        f=os.path.join(exp_dir, f"{model.__class__.__name__}_weights_{embedding_dim}.pth"),
    )

    # # Evaluate the model
    # model.evaluate_edge_predictions(
    #     dataloader=dataloader
    # )

    # embeddings = model.weight.cpu().detach().numpy()
    # fig, (ax1, ax2) = plt.subplots(1, 2)
    # ax1.plot(range(epochs + 100), losses)
    # ax2.scatter(x=embeddings[:, 0], y=embeddings[:, 1])
    # # plt.show()
    # fig.savefig(os.path.join(exp_dir, f"embeddings.png"))

    # node_order=[4, 30, 55, 72, 95, 34, 63, 64, 66, 75, 2, 11, 35, 46, 98, 36, 50, 65, 74, 80, 54, 62, 70, 82, 92, 0, 51, 53, 57, 83, 9, 10, 16, 28, 61, 22, 39, 40, 86, 87, 5, 20, 25, 84, 94, 6, 7, 14, 18, 24, 26, 45, 77, 79, 99, 1, 32, 67, 73, 91, 3, 42, 43, 88, 97, 15, 19, 21, 31, 38, 27, 29, 44, 78, 93, 8, 13, 48, 58, 90, 41, 69, 81, 85, 89, 12, 17, 37, 68, 76, 23, 33, 49, 60, 71, 47, 52, 56, 59, 96]
    # embeddings_dist_mat = create_dist_mat_plots(
    #     model.weight.cpu().detach(),
    #     node_permutation=node_order,
    #     ball=ball,
    #     output_dir=exp_dir,
    # )

    # graph_dist_mat = dataset.sampler.dist_matrix[node_order, :][:, node_order]
    # create_dist_diff_mat_plots(
    #     embeddings_dist_mat=embeddings_dist_mat,
    #     graph_dist_mat=graph_dist_mat,
    #     node_permutation=node_order,
    #     output_dir=exp_dir,
    # )
