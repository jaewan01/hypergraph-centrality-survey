import argparse
import time
import xgi
import numpy as np
import os
from xgi.exception import XGIError
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for XGI centrality algorithms.")
    parser.add_argument('--dataset', type=str, required=True, help="Specify the dataset to use.")
    parser.add_argument('--measure', type=str, required=True, help="Specify the centrality measure to test.")
    parser.add_argument('--edge', default=False, action='store_true', help="Specify if the test is for edge centrality.")
    args = parser.parse_args()

    hypergraph = xgi.load_xgi_data(args.dataset, read=True, path="../datasets_lcc_size_2")
    print("loaded hypergraph ", args.dataset)

    if not args.edge:
        time_start = time.time()
        if args.measure == "degree":
            centralities = hypergraph.nodes.degree_centrality.asnumpy()
        elif args.measure == "neighbor_degree":
            centralities = hypergraph.nodes.neighbor_degree_centrality.asnumpy()
        elif args.measure == "closeness":
            centralities = hypergraph.nodes.closeness_centrality.asnumpy()
        elif args.measure == "betweenness":
            centralities = hypergraph.nodes.betweenness_centrality.asnumpy()
        elif args.measure == "harmonic":
            centralities = hypergraph.nodes.harmonic_centrality.asnumpy()
        elif args.measure == "eigenvector":
            centralities = hypergraph.nodes.eigenvector_centrality(max_iter=5000, tol=1e-6).asnumpy()
        elif args.measure == "pagerank":
            centralities = hypergraph.nodes.pagerank_centrality(alpha=0.9, max_iter=5000, tol=1e-6).asnumpy()
        elif args.measure == "uplift_eigenvector":
            centralities = hypergraph.nodes.uplift_eigenvector_centrality(max_iter=5000, tol=1e-6).asnumpy()
        elif args.measure == "hypercoreness":
            centralities = hypergraph.nodes.hypercoreness.asnumpy()
        else:
            raise XGIError("Invalid centrality measure for node: ", args.measure)
        time_end = time.time()
        print(f"Time taken for node_{args.measure}: {time_end - time_start} seconds")
    else:
        time_start = time.time()
        if args.measure == "degree":
            centralities = hypergraph.edges.degree_centrality.asnumpy()
        elif args.measure == "line_expansion_degree":
            centralities = hypergraph.edges.line_expansion_degree_centrality.asnumpy()
        elif args.measure == "closeness":
            centralities = hypergraph.edges.closeness_centrality.asnumpy()
        elif args.measure == "betweenness":
            centralities = hypergraph.edges.betweenness_centrality.asnumpy()
        elif args.measure == "harmonic":
            centralities = hypergraph.edges.harmonic_centrality.asnumpy()
        elif args.measure == "eigenvector":
            centralities = hypergraph.edges.eigenvector_centrality(max_iter=5000, tol=1e-6).asnumpy()
        elif args.measure == "pagerank":
            centralities = hypergraph.edges.pagerank_centrality(alpha=0.9, max_iter=5000, tol=1e-6).asnumpy()
        elif args.measure == "hypercoreness":
            centralities = hypergraph.edges.hypercoreness.asnumpy()
        else:
            raise XGIError("Invalid centrality measure for edge: ", args.measure)
        time_end = time.time()
        print(f"Time taken for edge_{args.measure}: {time_end - time_start} seconds")

    tolerance = 1e-3
    assert np.sum(centralities) < 1 + tolerance
    assert np.sum(centralities) > 1 - tolerance

    if args.edge:
        measure_name = "edge_" + args.measure
    else:
        measure_name = "node_" + args.measure

    with open("runtime.txt", "a+") as f:
        f.write(f"{args.dataset} {measure_name} {time_end - time_start:.5f}s\n")

    os.makedirs(f"values/{args.dataset}/", exist_ok=True)
    np.save(f"values/{args.dataset}/{measure_name}.npy", centralities)
