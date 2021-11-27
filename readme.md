# LERP : Label-dependent and event-guided interpretable disease risk prediction using EHRs

## This is the code for the LERP.
![image](https://github.com/finnickniu/LERP/blob/main/IMG/LERP.png)
## Dataset

1. The dataset used is MIMIC-III, you should download it from https://physionet.org/content/mimiciii/1.4/ first.

2. Then, you should use MIMI-III benchmark tool to generate the Phenotype classification.

3. Next, you should use the py files under the folder of data_processing.

```
 python data_processing/generate_event.py 
 python data_processing/generate_text.py
 data_processing/split_data.py 
```
## Training 

After you generate the dataset, you could use:

``` 
    python trainer_text_event.py
```
to train the LERP

## Evaluation and case study

You could use the following commands to evaluate the LERP model and check the case study result.

```
    python evaluation_text_event.py
    python case_study_text_event.py
```
