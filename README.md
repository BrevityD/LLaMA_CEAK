# llama-ceak

Replace the upstream network of CEAK with the embedding layer of LLaMA.

Checkpoints are named by such schema:

llama-{1B}-{fd/od}-{p/n}-{f/uf}-{lr14}

where fd/od stands for filtered dataset (lce greater than 1) and origin dataset. 
p/n stands for pooling the embedding or not. 
f/uf stands for freezing the embedding layer weight or not. 
lr14 stands for learning rate with 1e-4, and so on.
