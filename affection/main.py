# main.py

from src.train import main as train_model
from src.tune import main as tune_model
from src.score import predict

if __name__ == "__main__":
    print("1. Train Model")
    print("2. Tune Model")
    print("3. Score Sample")

    choice = input("Select option: ")

    if choice == "1":
        train_model()
    elif choice == "2":
        tune_model()
    elif choice == "3":
        sample = [-58, -35, -36, 83.95, 0.339, 32.66]
        print("Prediction:", predict(sample))
    else:
        print("Invalid selection.")
