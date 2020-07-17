# Deep Learning for Automatic Seal Counting

## Progress 

### Done
 - Automated image cropping pipeline for different scales
 - Export and read images/examples through tfrecords
 - Implemented training and evaluation scripts
 - Started training for the baseline 416 input size models (trained models available under home/md273/model_zoo)
 - Load models from cfg files located in config/ directory
 
### IN-PROGRESS
  - Data augmentation pipeline (need more variety in underrepresented seal types)
  - Evaluator MAP bugs (Average MAP is incorrect)

### TODO
- Large image moving window evaluator.
    - Apply the new larger image models on the full images using a sliding window.
    - Evaluate if models accepting larger input images result in better overall predictions.
    
- 608, 1216, 1824, 2432 input models