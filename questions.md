# Questions

1. For these questions, extract and label the following metrics from your _m5out/stats.txt_ file for both the `lin_reg` and `lin_reg_iter` binaries:
    1. **(2 points)** The total number of shader active ticks
    2. **(2 points)** The percent utilization of the lanes in the GPU
    3. **(4 points)** The number of accesses the GPU made to the GPU memory
    4. **(2 points)** The number of ticks lost due to bank contention between thread blocks
    5. **(2 points)** The number of coalesced memory reads

2. Answer the following questions considering the source code of `lin_reg` and `lin_reg_iter`:
    1. **(2 points)** What factors contribute to the difference in the shader active ticks between the two workloads?
    2. **(2 points)** Warp divergence is a major contributor to a decrease in total utilization of GPU hardware. Explain what factors contribute to this from a code perspective, from a hardware perspective.
    3. **(4 points)** Coalesced memory reads/writes give GPUs a significant advantage in throughput when compared to CPUs. Why do we not implement memory access coalescing in CPUs? Why does this work so well in GPUs?

3. **(1 point)** The best time to be critical and to offer constructive suggestions about an assignment is soon after completing it. Please include the following in the report:
    1. A narrative description of any difficulties or misunderstandings that made the assignment unnecessarily difficult, or that led to wasted time.
    2. Suggestions of changes that could make this assignment more interesting, more relevant or more challenging in future editions of this course.


- (2 points) Proper typesetting of the report and overall presentation style. This includes the use of properly referenced figures, tables, and graphs (with descriptive axis titles and proper identification of units of measurement) where applicable.
- (2 points) Meeting notes. See the collaboration requirements for what to include.
- (1 point) Partner Evaluation Please complete the confidential partner evaluation for your partner. If you both complete the evaluation, you get a point.
