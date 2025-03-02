# K-Nearest Neighbors (KNN) Implementation

This repository contains an implementation of the **K-Nearest Neighbors (KNN) algorithm** from scratch using NumPy. The implementation supports both **classification** and **regression** tasks and allows users to choose between **Euclidean** and **Manhattan** distance metrics.

## Features
- Supports **both classification and regression** tasks.
- Implements **Euclidean** and **Manhattan** distance metrics.
- Uses **NumPy** for efficient numerical computations.
- Simple and modular structure with a `BaseModel` inheritance.
---

## Class: `KNearestNeighbour`

### Constructor

```python
def __init__(
    self,
    num_neighbours: int = 3,
    dist_func_type: Literal["eucledian", "manhattan"] = "eucledian",
    task: Literal["classification", "regression"] = "classification",
)
```

#### Parameters:
- `num_neighbours` (**int**): Number of neighbors to consider (default: `3`).
- `dist_func_type` (**str**): Distance metric to use (`"eucledian"` or `"manhattan"`, default: `"eucledian"`).
- `task` (**str**): Specifies whether the model is for **classification** or **regression** (default: `"classification"`).

---

### Methods

#### `forward(x: np.ndarray, y: np.ndarray)`
Stores the training data for KNN.

- `x`: Training features (`numpy.ndarray`).
- `y`: Corresponding labels (`numpy.ndarray`).

---

#### `predict(x_test: np.ndarray) -> np.ndarray`
Predicts labels or values for test data.

- `x_test`: Test data (`numpy.ndarray`).
- **Returns**: Predicted labels (for classification) or predicted values (for regression).

---

#### Private Methods:
- `__get_label(x: np.ndarray)`: Predicts class label for a given sample.
- `__get_value(x: np.ndarray)`: Predicts continuous value (for regression).
- Both methods compute distances using `Distant` class.
---

## Dependencies
- `numpy`

To install NumPy:
```bash
pip install numpy
```

---