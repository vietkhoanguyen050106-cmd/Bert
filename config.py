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
    """mode is 'baseline' (hard labels only) or 'distill' (learns from the teacher)."""
    return os.path.join(OUTPUT_ROOT, f"student_{mode}_{num_layers}L")
 
 
def is_trained(directory):
    """A model counts as finished once save_model() wrote config.json in its folder."""
    return os.path.exists(os.path.join(directory, "config.json"))