# llama-ceak

Replace the upstream network of CEAK with the embedding layer of LLaMA.

This repository does not have a stable release and its updates have stalled. Users are advised to be cautious.

**Notice**: training a few layers of LlaMA is only supported with `transformers<=4.46`, as the `forward` method of a single layer no longer accepts `position_id` parameter. In `transformers>4.46`, the `forward` method receives the `position_embedding` parameter instead. Please modify the code accordingly if you use `transformers>4.46`.

Checkpoints are named by such schema:

llama-{1B}-{fd/od}-{p/n}-{f/uf}-{lr14}

where fd/od stands for filtered dataset (lce greater than 1) and origin dataset. 
p/n stands for pooling the embedding or not. 
f/uf stands for freezing the embedding layer weight or not. 
lr14 stands for learning rate with 1e-4, and so on.
