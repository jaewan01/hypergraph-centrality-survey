cd centrality

datasets=("senate-bills" "house-bills" "email-enron" "email-eu" "contact-primary-school" "contact-high-school" "tags-ask-ubuntu" "threads-ask-ubuntu" "ndc-classes" "ndc-substances")

node_centralities=("degree" "neighbor_degree" "closeness" "betweenness" "harmonic" "eigenvector" "pagerank" "uplift_eigenvector" "hypercoreness")
edge_centralities=("degree" "line_expansion_degree" "closeness" "betweenness" "harmonic" "eigenvector" "pagerank" "hypercoreness")

# Loop over datasets
for dataset in "${datasets[@]}"; do
    
    # Loop over node centralities (uncomment if needed)
    for centrality in "${node_centralities[@]}"; do
        python test_single_centrality.py --dataset "$dataset" --measure "$centrality"
    done

    # Loop over edge centralities
    for centrality in "${edge_centralities[@]}"; do
        python test_single_centrality.py --dataset "$dataset" --measure "$centrality" --edge
    done

done
