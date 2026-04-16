# References

A short, practitioner-oriented list of the resources I actually consulted
while building this project. Not a textbook bibliography.

## Methodology & algorithms

1. **Ke, T., et al. (2017).** *LightGBM: A Highly Efficient Gradient
   Boosting Decision Tree.* NeurIPS. Background on histogram-based
   gradient boosting, which is the architecture behind
   `sklearn.ensemble.HistGradientBoostingClassifier`.
2. **Chen, T. & Guestrin, C. (2016).** *XGBoost: A Scalable Tree
   Boosting System.* KDD. Methodological baseline and the canonical
   reference for sample-weighted gradient boosting.
3. **Huang, Z. (1998).** *Extensions to the k-Means Algorithm for
   Clustering Large Data Sets with Categorical Values.* Data Mining
   and Knowledge Discovery 2(3). The original K-Prototypes paper.
4. **Saito, T. & Rehmsmeier, M. (2015).** *The precision-recall plot
   is more informative than the ROC plot when evaluating binary
   classifiers on imbalanced datasets.* PLoS ONE 10(3). Motivation
   for treating PR-AUC as the headline metric at a 6% base rate.
5. **Niculescu-Mizil, A. & Caruana, R. (2005).** *Predicting Good
   Probabilities With Supervised Learning.* ICML. Why gradient
   boosting with `log_loss` is already well-calibrated out of the
   box — matches the empirical result in §5 of the report.
6. **Barocas, S., Hardt, M., Selbst, A. (2019).** *Fairness and
   Machine Learning: Limitations and Opportunities.* Framework for
   the fairness discussion in §7.

## Data & dataset notes

7. **U.S. Census Bureau.** *Current Population Survey Technical
   Documentation* (CPS ASEC, 1994 and 1995 supplement files).
   Used for: the definition of the survey weight, the "not in
   universe" category semantics, and the top-coding rules on
   capital-gains income ($99,999 is the top-coded value in the
   dataset used in this project).
8. **KDD Cup 1999.** *Census-Income (KDD) Data Set.* Available
   via the UCI ML Repository. The 199,523-row file shipped with
   this project is the "train" portion of that release.

## Libraries & tools

9. **Pedregosa et al. (2011).** *scikit-learn: Machine Learning in
   Python.* JMLR. Primary library. Specific docs consulted:
   `HistGradientBoostingClassifier` (categorical features, sample
   weights, early stopping); `calibration_curve`;
   `StratifiedKFold`; `permutation_importance` (for comparison
   with our weighted re-implementation).
10. **de Vos, N. J.** *kmodes: Python implementations of the
    k-modes and k-prototypes clustering algorithms.*
    https://github.com/nicodv/kmodes
11. **Akiba, T. et al. (2019).** *Optuna: A Next-generation
    Hyperparameter Optimization Framework.* KDD. Used for the
    classifier hyperparameter search with the TPE sampler and
    median pruner.

## Practitioner references

12. **Kuhn, M. & Silge, J. (2023).** *Tidy Modeling with R.* Chapter
    11 ("Comparing Models with Resampling") — the discussion on
    picking thresholds on validation data rather than on test was
    my mental model for §4.3.
13. **Molnar, C. (2022).** *Interpretable Machine Learning.* Chapter
    on permutation importance — methodology for the weighted
    permutation-importance implementation in
    `src/census_income/classifier.py`.
14. **Google PAIR.** *People + AI Guidebook — "Mental Models"*
    chapter. Framing for the §5.5 discussion on how the classifier
    and the segmentation compose for marketing use.
