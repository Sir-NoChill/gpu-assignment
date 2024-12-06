# Background

//TODO justify the assignment, give context for why GPUs are important

## Objective

Recall Dr. Amaral's lectures into the architecture of GPU systems and how different they are from CPU systems. You should be familiar with how the lectures introduce the compute units of the GPU and their organization within the larger GPU system, as well as the architectural features of a GPU's L1, L2 and GDDR Memory. The goal of this lab is to further expand your understanding of GPU architecture and its effects on performance by having you modify certain elements of a GPU and observing the effects on training a simple neural network.

## GPU's in gem5

GPU support in the gem5 simulator is not as widely understood as the CPU support. The lab at UW Madison performed extensive work on including GPU systems into gem5 in the early 2020's and the work is now contained in the upstream repository. The current architecture allows for full simulation of the AMD ROCm stack as well as AMD GPU systems (as of writing, up to the MI3000). 

GPU simulation in the gem5 can either be done entirely in simulation or by using a combination of simulated and host resources. As you can imagine, full simulation of CPU systems with discrete GPGPU's and their workloads would be time consuming, so gem5 has additional support for allowing the host system to perform workloads using the KVM ISA extension for x86 CPU's in order to decrease execution time. Our simulations will take advantage of this capability in order to simulate training a classifier for the MNIST dataset on a simulated AMD GPU.

## GPU Cores

Since we are studying AMD systems, we will use AMD terminology. Below is a table that compares the AMD terms with the NVIDIA terms for those with familiarity in the NVIDIA CUDA stack:

| **Component**        | **NVIDIA Term**               | **AMD Term**                   |
|----------------------|--------------------------------|--------------------------------|
| Processing Unit      | Streaming Multiprocessor (SM) | Compute Unit (CU)              |
| Execution Group      | Warp (32 threads)             | Wavefront (64 threads)         |
| Processing Core      | CUDA Core                     | SIMD Lane or Processing Element|
| Thread Group         | Thread Block                  | Workgroup                      |
| Individual Thread    | Thread                        | Thread                         |

The term 'thread' might be a bit misleading in the context of a GPU, because a thread is significantly different on a GPU than a CPU relative to the software interface we use to program it. Instead of executing programs as a sequence of _individual_ instructions, GPU's execute a Workgroup. So a collection of individual threads, sometimes taken from different wavefronts, are done in parallel. The caveat of this system of execution is that every thread needs to be performing the same operation.

It falls to the Processing element to perform this operation. The Processing element is not dissimilar to your vector ALU on a CPU. It will consume a vector (or two) of inputs and then perform an operation on them. In fact, in more modern GPU systems, sometimes there are entire processing elements dedicated to matrix multiplication (see NVIDIA's tensor cores and spare tensor cores [see here](LIIIINK).


