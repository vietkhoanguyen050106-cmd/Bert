
MODEL_NAME = "bert-base-uncased"  
NUM_LABELS = 2                     
NUM_STUDENT_LAYERS = 6           
MAX_LENGTH = 128               
SEED = 42                          

TEACHER_DIR = "./outputs/teacher"
STUDENT_BASELINE_DIR = "./outputs/student_baseline"
STUDENT_DISTILL_DIR = "./outputs/student_distilled"


TEMPERATURE = 2.0  
ALPHA = 0.5        