## Project to try to solve rubik cube with model based reinforcement learning

The algorithm idea is this :

![image](https://github.com/user-attachments/assets/22b5fc00-7955-423f-9d4e-8676a5b44906)

## First experience 

Firstly we try with a mamba neural architecture and continuous endpoint parametrization. Not working great (although it gives good results). 

I also think that we should have a better "reward generation" algorithm.

To generate a reward value we can simply diffuse toward the end state (like in diffusion painting).

## Second experience

In the second experience we will simply use a transformer instead of a mamba model.

Also we will go from continuous loss toward discrete one like in the paper : https://arxiv.org/pdf/2404.19739v1 

- One big point is that when using film layer to include scalar input, then use (1. + element) for multiplication set (to avoid bad conditioning)

## Third experience 

Keep the transformer / soap optimizer setup but used a cross entropy loss instead
