
# Learning Rate Schedulers

A **Learning Rate Scheduler (LRScheduler)** is used to adjust the learning rate dynamically during training. A well-designed scheduler helps in faster convergence and better generalization.

---

##  Available Learning Rate Schedulers

### 1️⃣ **Constant Learning Rate**
- **Definition**: Keeps the learning rate constant throughout training.
- **Formula**:  
  \[
  LR_t = LR_0
  \]
- **Use Case**:  
  - Useful when training from scratch and a stable learning rate is required.

---

### 2️⃣ **Step Learning Rate**
- **Definition**: Reduces the learning rate by a factor (`gamma`) every `step_size` epochs.
- **Formula**:  
  \[
  LR_t = LR_0 \times \gamma^{\lfloor t / step\_size \rfloor}
  \]
- **Use Case**:  
  - Suitable for scenarios where the model initially needs a higher learning rate, then reduces it in steps to fine-tune performance.

---

### 3️⃣ **Exponential Decay Learning Rate**
- **Definition**: Reduces the learning rate by a constant factor (`gamma`) at each step.
- **Formula**:  
  \[
  LR_t = LR_0 \times e^{-\gamma t}
  \]
- **Use Case**:  
  - Ideal for cases where a smooth and continuous learning rate decay is required.

---

### 4️⃣ **Linear Decay Learning Rate**
- **Definition**: Linearly decreases the learning rate from `initial_lr` to `final_lr` over `total_steps`.
- **Formula**:  
  \[
  LR_t = LR_0 \times \left(1 - \frac{t}{total\_steps}\right)
  \]
- **Use Case**:  
  - Used when training needs a gradual reduction of learning rate.

---

### 5️⃣ **Cosine Annealing Learning Rate**
- **Definition**: Uses a cosine function to decay the learning rate.
- **Formula**:  
  \[
  LR_t = \frac{LR_0}{2} \left(1 + \cos\left(\frac{\pi t}{total\_steps}\right)\right)
  \]
- **Use Case**:  
  - Used for smooth decay in deep learning tasks, especially in image classification.

---

### 6️⃣ **Cyclical Learning Rate**
- **Definition**: Alternates learning rate between `base_lr` and `max_lr` in a cyclic manner.
- **Formula**:  
  \[
  LR_t = base\_lr + \frac{(max\_lr - base\_lr)}{2} \left(1 + \cos\left(\frac{\pi t}{cycle\_length}\right)\right)
  \]
- **Use Case**:  
  - Useful when training a model where exploration and exploitation are needed in different phases.

---

### 7️⃣ **Polynomial Decay Learning Rate**
- **Definition**: Reduces the learning rate using a polynomial function.
- **Formula**:  
  \[
  LR_t = LR_0 \times \left(1 - \frac{t}{total\_steps}\right)^{power}
  \]
- **Use Case**:  
  - Used in large-scale training tasks where steady decay is necessary.

---

### 8️⃣ **Reduce on Plateau Learning Rate**
- **Definition**: Reduces the learning rate when validation loss stops improving.
- **Formula**:  
  If validation loss has not improved for `patience` epochs:
  \[
  LR_t = LR_0 \times \gamma
  \]
- **Use Case**:  
  - Works well for adaptive training scenarios where manual tuning is difficult.

---

## How to Use These Schedulers
```python
optimizer = AdamOptimizer(lr=0.001)
scheduler = LRScheduler(optimizer, method="cosine", total_steps=100)

for epoch in range(epochs):
    train_model()
    optimizer.step()
    scheduler.step() 
```
