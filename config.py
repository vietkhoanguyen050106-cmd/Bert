import os
 

MODEL_NAME = "bert-base-uncased"   
NUM_LABELS = 2                   
MAX_LENGTH = 128                   
SEED = 42                         
 

NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 32
 

TEMPERATURE = 2.0   
ALPHA = 0.5         
BETA = 0.0         
 
OUTPUT_ROOT = "./outputs"
TEACHER_DIR = os.path.join(OUTPUT_ROOT, "teacher")
 
 
def student_dir(mode, num_layers):
   
    return os.path.join(OUTPUT_ROOT, f"student_{mode}_{num_layers}L")
 
 
def is_trained(directory):
   
    return os.path.exists(os.path.join(directory, "config.json"))