Data Drift Analysis

1. Which features showed drift and why?
The features Age and DailyRate showed significant drift. This occurred because synthetic drift was manually injected into the production dataset by increasing the Age values and scaling the DailyRate values. This shift was correctly identified by the Evidently drift detection report.

2. Would this drift likely affect model performance?
Yes. Since the model was trained on a specific distribution of employee demographics and compensation, a systemic shift in these features means the model is making predictions on data it no longer recognizes accurately. This would likely lead to a drop in precision and recall.

3. What action would you recommend?
I recommend investigating the data source to confirm if this shift reflects a real-world change in hiring or a data pipeline error. If the change is legitimate, the model must be retrained using the updated data distribution to restore performance.