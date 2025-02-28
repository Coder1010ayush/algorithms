# Random Forest Implementation Details

## 1. Decision Tree Variants Used
I implemented Random Forest using three different decision tree variants:
- **DecisionTreeID3**: Uses information gain for splitting (entropy-based criterion).
- **DecisionTreeCART**: Supports both classification (Gini index) and regression (MSE criterion).
- **DecisionTreeRegression**: Specifically optimized for regression tasks using MSE.

## 2. Bootstrap Sampling
- Each tree in the forest is trained on a random subset (with replacement) of the training data.
- The size of the subset is equal to the original training set but sampled with replacement.

## 3. Feature Selection per Split
- Instead of using all features, a random subset of features is selected at each split.
- Classification: Defaults to `sqrt(n_features)`.
- Regression: Defaults to `n_features / 3`.
- Users can dynamically adjust this through hyperparameters, allowing for cross-validation optimization.

## 4. Out-of-Bag (OOB) Error Estimation
- For each sample, track trees that did not include it in their training data (OOB trees).
- Compute predictions using only OOB trees and compare them to actual labels to estimate the model's generalization error.

## 5. Weighted Voting for Classification
- Instead of majority voting, trees can contribute votes weighted by confidence scores.
- Uses class probabilities from individual trees to refine the ensemble decision.

## 6. Regularization and Pruning
- Added pruning mechanisms to **DecisionTreeID3** and **DecisionTreeCART**:
  - **Minimum Gain Threshold**: Prevents splits that do not provide sufficient information gain.
  - **Maximum Depth**: Limits tree depth to prevent overfitting.
  - **Minimum Sample Split**: Ensures a minimum number of samples per split.

## 7. Parallelization (Optional)
- Introduced parallel training using `joblib`.
- Users can toggle parallel execution for training trees to speed up computation.

## 8. Random Forest Hyperparameters
- **n_trees**: Number of trees in the ensemble.
- **max_features**: Number of features considered for each split.
- **max_depth**: Maximum depth for individual trees.
- **min_samples_split**: Minimum number of samples required for a split.
- **bootstrap**: Whether to use bootstrap sampling.
- **oob_score**: Whether to compute OOB error.
- **n_jobs**: Number of parallel jobs (optional parallel execution).

## 9. Prediction Process
- **Classification**: Aggregates tree predictions using majority voting or weighted voting.
- **Regression**: Averages predictions from all trees to get the final output.
