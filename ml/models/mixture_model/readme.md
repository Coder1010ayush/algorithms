# **Gaussian Mixture Model (GMM) & Hidden Markov Model (HMM)**

## **1️⃣ Gaussian Mixture Model (GMM)**

### **Overview**
A **Gaussian Mixture Model (GMM)** is a **probabilistic model** used for **clustering** and **density estimation**. It assumes that the data is generated from a mixture of multiple **Gaussian distributions**.

### **Mathematical Formulation**
A GMM is defined as:

\[
P(X) = \sum_{i=1}^{K} \pi_i \mathcal{N}(X | \mu_i, \Sigma_i)
\]

where:
- \( K \) = number of Gaussian components (clusters)
- \( \pi_i \) = **mixing coefficient** (probability of selecting component \( i \)), with \( \sum_{i=1}^{K} \pi_i = 1 \)
- \( \mathcal{N}(X | \mu_i, \Sigma_i) \) = **multivariate normal distribution** with:
  - **Mean** \( \mu_i \)
  - **Covariance matrix** \( \Sigma_i \)

### **Expectation-Maximization (EM) Algorithm**
GMM is trained using the **EM algorithm**, which iteratively updates parameters until convergence.

#### **Step 1: Initialization**
- Initialize means \( \mu_i \), covariances \( \Sigma_i \), and mixing coefficients \( \pi_i \).
- Scikit-learn supports different initialization methods: `"kmeans"`, `"random"`, `"random_from_data"`.

#### **Step 2: Expectation Step (E-Step)**
- Compute the probability (responsibilities) of each data point belonging to each Gaussian:

\[
r_{i,j} = \frac{\pi_i \mathcal{N}(x_j | \mu_i, \Sigma_i)}{\sum_{k=1}^{K} \pi_k \mathcal{N}(x_j | \mu_k, \Sigma_k)}
\]

#### **Step 3: Maximization Step (M-Step)**
- Update parameters using weighted averages:

\[
\mu_i^{\text{new}} = \frac{\sum_{j=1}^{N} r_{i,j} x_j}{\sum_{j=1}^{N} r_{i,j}}
\]

\[
\Sigma_i^{\text{new}} = \frac{\sum_{j=1}^{N} r_{i,j} (x_j - \mu_i)(x_j - \mu_i)^T}{\sum_{j=1}^{N} r_{i,j}}
\]

\[
\pi_i^{\text{new}} = \frac{\sum_{j=1}^{N} r_{i,j}}{N}
\]

#### **Step 4: Check for Convergence**
- Stop when log-likelihood improvement is below a threshold.

---

## **2️⃣ Hidden Markov Model (HMM)**

### **Overview**
A **Hidden Markov Model (HMM)** is a **probabilistic model** used to represent **sequential data**. It assumes:
1. There is an **underlying Markov process** with **hidden (latent) states**.
2. Each state generates an **observable output** based on an **emission probability distribution**.

### **Mathematical Components**
An HMM consists of:
1. **States**: \( S = \{ S_1, S_2, ..., S_N \} \) (hidden)
2. **Observations**: \( O = \{ o_1, o_2, ..., o_T \} \) (observable)
3. **Transition Probabilities**: \( A = P(S_t | S_{t-1}) \)
4. **Emission Probabilities**: \( B = P(O_t | S_t) \)
5. **Initial Probabilities**: \( \pi = P(S_1) \)

### **Mathematical Representation**
- **State Transition Matrix** \( A \):

\[
A_{ij} = P(S_t = S_j | S_{t-1} = S_i)
\]

- **Emission Matrix** \( B \):

\[
B_{ij} = P(O_t | S_t)
\]

- **Initial State Distribution** \( \pi \):

\[
\pi_i = P(S_1 = S_i)
\]

### **Key HMM Algorithms**
HMMs rely on three main algorithms:

#### **1️⃣ Forward Algorithm (Probability Estimation)**
Used to compute the probability of an observation sequence:

\[
\alpha_t(j) = P(O_1, O_2, ..., O_t, S_t = S_j)
\]

#### **2️⃣ Viterbi Algorithm (Most Likely Sequence)**
Finds the most probable sequence of hidden states:

\[
\delta_t(j) = \max_{S_{t-1}} P(O_1, ..., O_t, S_t = S_j)
\]

#### **3️⃣ Baum-Welch Algorithm (Parameter Learning)**
An EM-based algorithm to **train HMM parameters** from data.

---