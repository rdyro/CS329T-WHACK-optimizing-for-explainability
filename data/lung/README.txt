Information about the LIDC/IDRI dataset and its license is found at [https://luna16.grand-challenge.org/Data/](https://luna16.grand-challenge.org/Data/).

For ease of reference, following is the data structure:
- subset0.zip to subset9.zip: 10 zip files which contain all CT images. NOT included. Download from [https://zenodo.org/record/3723295#.YpTG4ajMJPY](https://zenodo.org/record/3723295#.YpTG4ajMJPY).
- annotations.csv: csv file that contains the annotations used as reference standard for the 'nodule detection' track
- sampleSubmission.csv: an example of a submission file in the correct format
- candidates.csv: the original set of candidates used for the LUNA16 workshop at ISBI2016. This file is kept for completeness, but should not be used, use candidates_V2.csv instead (see more info below).
- candidates_V2.csv: csv file that contains an extended set of candidate locations for the ‘false positive reduction’ track. 
- evaluation script: the evaluation script that is used in the LUNA16 framework
- lung segmentation: a directory that contains the lung segmentation for CT images computed using automatic algorithms. NOT included. Download the zipped version seg-lungs-LUNA16.zip from [https://zenodo.org/record/3723295#.YpTG4ajMJPY](https://zenodo.org/record/3723295#.YpTG4ajMJPY).
- additional_annotations.csv: csv file that contain additional nodule annotations from our observer study. The file will be available soon. This file is not included in this repository.