# Activity Role Classification

To run this approach, the event logs must be placed in `./inputs/logs/`:
- `BPI Challenge 2017.xes`
- `BPI Challenge 2018.xes`
- `BPI_Challenge_2012.xes`
- `BPI_Challenge_2019.xes`
- `BPIC15_1.xes`
- `CCC19 - Log CSV.csv`
- `CreditRequirement.csv`
- `Detail_Incident_Activity.csv`
- `Hospital Billing - Event Log.xes`
- `PermitLog.xes`
- `Receipt phase of an environmental permit application process (‘WABO’), CoSeLoG project.xes`
- `Road_Traffic_Fine_Management_Process.xes`
- `Sepsis Cases - Event Log.xes`

The logs can be downloaded here:

| Log | Link |
| ------ | ------ |
| BPI Challenge '12 | https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f |
| BPI Challenge '13 closed incidents | https://doi.org/10.4121/uuid:c2c3b154-ab26-4b31-a0e8-8f2350ddac11 |
| BPI Challenge '14 Acitivity log | https://doi.org/10.4121/uuid:86977bac-f874-49cf-8337-80f26bf5d2ef | 
| BPI Challenge '15 Municipality 1 | https://doi.org/10.4121/uuid:a0addfda-2044-4541-a450-fdcc9fe16d17 | 
| BPI Challenge '17 | https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b | 
| BPI Challenge '18 | https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972 | 
| BPI Challenge '19 | https://doi.org/10.4121/uuid:d06aff4b-79f0-45e6-8ec8-e19730c248f1 | 
| BPI Challenge '20 Permit log | https://doi.org/10.4121/uuid:ea03d361-a7cd-4f5e-83d8-5fbdf0362550 | 
| Conformance Checking Challenge '19 | https://doi.org/10.4121/uuid:c923af09-ce93-44c3-ace0-c5508cf103ad | 
| Credit Requirements | https://doi.org/10.4121/uuid:453e8ad1-4df0-4511-a916-93f46a37a1b5 | 
| Hospital Billing | https://doi.org/10.4121/uuid:76c46b83-c930-4798-a1c9-4be94dfeb741 | 
| Road Traffic Fine Management | https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5 | 
| Sepsis cases | https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460 | 
| WABO Receipt phase | https://doi.org/10.4121/uuid:26aba40d-8b2d-435b-b5af-6d4bfbd7a270 | 



# Installation of Semantic Component Extraction from Event Data
1. Install via pip: <code>pip install git+https://gitlab.uni-mannheim.de/processanalytics/semantic-role-extraction.git</code>
2. Download the models used by the package <code>python -m extraction download role_extraction_models</code>. The following is downloaded:<br>
    <small> A [spacy](https://spacy.io) model (en_core_web_md) used for part-of-speech tagging<br>
 A fine-tuned [BERT](https://github.com/google-research/bert) model to extract roles from textual values (event attributes)<br>
 [GloVe embeddings](https://nlp.stanford.edu/projects/glove/) (glove.6B.100d) for determining semantic similarity between textual values.</small>


# Installation of Requirements
Install the dependencies in requirements.txt: using pip <code> pip install -r requirements.txt </code>

# Run Feature Extraction Approach

1. Run `./src/preprocessing/log_augmentation.py`. Some conflicts in python packages might occur which need to be resolved.
2. Run `./src/preprocess/log_preprocessing.py`.
3. Run `./src/feature_extraction/extract_features.py`

# Train / Evaluate Classifiers

Run the python files in `./src/classification/classifier/` for the desired classifier.
If the predictions are already in the output log, it will only evaluate and not retrain.
Remove the files to retrain the classifiers.

