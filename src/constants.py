import os

RANDOM_SEED = 42

# Extracted semantic components
SEMANTIC_ROLES = [
    'object:name',
    'object:status',
    'action:name',
    'action:status',
    'org:actor:name',
    'org:passive:name',
    'resource:type',
]

# Categorization scheme
ANNOTATION_LABELS = ['transform', 'decide', 'assess', 'create', 'move', 'communicate', 'preserve', 'destroy',
                     'combine', 'separate', 'manage']

# Default keys
DEFAULT_ACTIVITY_KEY = 'concept:name'
DEFAULT_CASE_KEY = 'case:' + DEFAULT_ACTIVITY_KEY
DEFAULT_EVENT_TIME_KEY = 'time:timestamp'
ANNOTATION_KEY = 'annotation'
EVENT_LOG_NAME_KEY = 'event:log:name'

# Special keys for some logs
KNOWN_LOG_KEYS = {'Detail_Incident_Activity': 'Incident ID',
                  'CreditRequirement': 'case',
                  'CCC19 - Log CSV': 'CASEID'}
KNOWN_EVENT_TIME_KEYS = {'Detail_Incident_Activity': 'DateStamp',
                         'CreditRequirement': 'startTime',
                         'CCC19 - Log CSV': 'START'}
KNOWN_ACTIVITY_KEYS = {'Detail_Incident_Activity': 'IncidentActivity_Type',
                       'CreditRequirement': 'event',
                       'CCC19 - Log CSV': 'ACTIVITY',
                       'BPI Challenge 2018': 'activity',
                       'BPIC15_1': 'activityNameEN'}
KNOWN_SEPARATORS = {'Detail_Incident_Activity': ';',
                    'CreditRequirement': ',',
                    'CCC19 - Log CSV': ','}


# Paths
FILEPATH_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Input paths
FILEPATH_LOGS = os.path.join(FILEPATH_PROJECT_ROOT, '../inputs/logs/')
FILEPATH_ANNOTATIONS = os.path.join(FILEPATH_PROJECT_ROOT, '../inputs/annotations/')
FILEPATH_OTHER = os.path.join(FILEPATH_PROJECT_ROOT, '../inputs/other/')

# Input files
FILE_ACTIVITY_ROLE_ANNOTATIONS = os.path.join(FILEPATH_ANNOTATIONS, 'activity_roles_annotations.csv')

ALL_EVENT_LOGS = os.listdir(FILEPATH_LOGS)
ALL_EVENT_LOGS = [f for f in ALL_EVENT_LOGS if f.endswith('.xes') or f.endswith('.csv')]
ALL_EVENT_LOG_NAMES = [log_name[:-4] for log_name in ALL_EVENT_LOGS]

# Output paths
FILEPATH_PREPROCESSING = os.path.join(FILEPATH_PROJECT_ROOT, '../outputs/preprocessing/')
FILEPATH_AUGMENTED = os.path.join(FILEPATH_PREPROCESSING, './augmented_logs/')
FILEPATH_PREPROCESSED = os.path.join(FILEPATH_PREPROCESSING, './preprocessed_logs/')
FILEPATH_FEATURES = os.path.join(FILEPATH_PREPROCESSING, './features/')

FILEPATH_MODELS = os.path.join(FILEPATH_PROJECT_ROOT, '../outputs/models/')
FILEPATH_MODELS_BERT = os.path.join(FILEPATH_MODELS, './bert/')

FILEPATH_RESULTS = os.path.join(FILEPATH_PROJECT_ROOT, '../outputs/results/')
FILEPATH_GRIDSEARCH = os.path.join(FILEPATH_RESULTS, './gridsearch/')
FILEPATH_PREDICTIONS = os.path.join(FILEPATH_RESULTS, './predictions/')
FILEPATH_FIGURES = os.path.join(FILEPATH_RESULTS, './figures/')

FILEPATH_USECASE = os.path.join(FILEPATH_PROJECT_ROOT, '../outputs/usecase_analysis')
