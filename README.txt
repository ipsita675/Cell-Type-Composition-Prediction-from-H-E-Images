
Project: Cell-Type Prediction from H&E Images

Pipeline:
1. Base model: ResNet18 CNN
2. Second stage: ElasticNet refinement using OOF predictions
3. Optional post-processing: Quantile calibration

Contents:
- models/: trained regression models
- predictions/: OOF + test predictions
- figures/: plots used in report
- configs/: metadata and setup info

Notes:
- GCN models were explored but underperformed
- Multi-scale CNN improves base predictions
- Structured refinement improves consistency
