# Distributed Fair k-Center Clustering with PySpark

A distributed implementation of the Fair Farthest-First Traversal (FairFFT) algorithm for the $k$-center clustering problem under group fairness constraints, built on Apache Spark.

---

### Overview

Standard $k$-center clustering algorithms (such as Gonzalez's Farthest-First Traversal) select cluster centers based purely on geometric distance, often leading to under-representation of sensitive protected groups. 

This project implements a two-round MapReduce framework (**MRFairFFT**) to solve the $k$-center problem with exact demographic representation constraints:
* Points belong to demographic groups (e.g., Label `A` or `B`).
* The user specifies the exact quota of centers to extract from each group ($k_A$ and $k_B$).
* The pipeline scales horizontally across Spark partitions using a coreset-based approach, bounding the maximum clustering radius while preserving fairness guarantees.

---

### Key Components

* **`FairFFT` (Sequential Algorithm):** An adapted Farthest-First Traversal heuristic that enforces group capacities ($k_A, k_B$). It greedily picks the farthest point from the current set of centers that does not violate the remaining group quotas.
* **`MRFairFFT` (Distributed MapReduce):**
  * **Round 1 (Local Coreset Extraction):** Spark partitions run `FairFFT` in parallel via `mapPartitions`, extracting $2k_A$ and $2k_B$ candidate representatives per partition.
  * **Round 2 (Global Consolidation):** The driver collects all local candidate centers and runs a final `FairFFT` pass to extract the exact $k_A + k_B$ centers.
* **`objective_funct`:** Computes the global $k$-center objective ($r(S) = \max_{u \in U} \min_{c \in S} dist(u, c)$) in parallel across the entire dataset.

---

### Input Data Format

The script expects a comma-separated values (CSV) file without headers. Each row represents a multi-dimensional point where all entries except the last are numerical coordinates, and the final entry is the group label:

```text
1.23,4.56,7.89,A
0.45,2.11,3.32,B
9.10,1.15,0.02,A
```

### Requirements

* Python 3.8+
* Apache Spark (PySpark) 3.x

Install dependencies:
```bash
pip install pyspark
```

### Execution

Run the script via `spark-submit`:

```bash
spark-submit main.py <file_path> <kA> <kB> <L>
```

### Arguments
* ⁠<file_path>⁠: Path to the input dataset file.
* ⁠<kA>⁠: Number of centers required from group ⁠A⁠.
* ⁠<kB>⁠: Number of centers required from group ⁠B⁠.
* ⁠<L>⁠: Number of Spark partitions.


### Sample Output
* File path = points.csv, KA = 5, KB = 5, L = 16
* N = 100000, NA = 60000, NB = 40000
* Center = [12.4,5.1,1.9] Label = A
* Center = [-2.1,8.3,0.4] Label = B
...
* Objective function = 3.482104
* Running time of MRFairFFT = 1420 ms
