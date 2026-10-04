import argparse
import subprocess
import sys
 
from Bert.config import NUM_EPOCHS, TEACHER_DIR, is_trained, student_dir
 
 
def run(module, *extra):
    command = [sys.executable, "-m", module, *extra]
    print("\n>>>", " ".join(command), flush=True)
    subprocess.run(command, check=True)     
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6])
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    args = parser.parse_args()
 
    if not is_trained(TEACHER_DIR):
        run("Bert.train_teacher")
 
    for layers in args.layers:
        extra = ["--layers", str(layers), "--epochs", str(args.epochs)]
        if not is_trained(student_dir("baseline", layers)):
            run("Bert.train_student", *extra)
        if not is_trained(student_dir("distill", layers)):
            run("Bert.train_distill", *extra)
 
    run("Bert.benchmark")
 
 
if __name__ == "__main__":
    main()