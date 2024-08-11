import os

def main():
    print("Welcome to Curvetopia: A Journey into the World of Curves")
    print("This project involves processing 2D curves from polylines to cubic Bézier curves,")
    print("with tasks that include regularizing curves, exploring symmetry, and completing incomplete curves.")
    print("\n")

    # Run data preparation script
    data_preparation_path = 'training_data/data_preparation/data_preparation.py'
    if os.path.exists(data_preparation_path):
        os.system(f'python {data_preparation_path}')
    else:
        print(f"Error: Could not find {data_preparation_path}")
        return

    # Ask to train
    user_input = input("Press 't' to train: ")
    if user_input.lower() == 't':
        train_path = 'training_data/train.py'
        if os.path.exists(train_path):
            os.system(f'python {train_path}')
        else:
            print(f"Error: Could not find {train_path}")
            return
    else:
        print("Exiting program.")
        return

    # Ask to regularize
    user_input = input("Press 'r' to regularize: ")
    if user_input.lower() == 'r':
        regularize_path = 'regularize.py'
        if os.path.exists(regularize_path):
            os.system(f'python {regularize_path}')
        else:
            print(f"Error: Could not find {regularize_path}")
            return
    else:
        print("Exiting program.")
        return

    # Ask to find symmetry
    user_input = input("Press 's' to find symmetry: ")
    if user_input.lower() == 's':
        symmetry_path = 'linesOfSymmetry.py'
        if os.path.exists(symmetry_path):
            os.system(f'python {symmetry_path}')
        else:
            print(f"Error: Could not find {symmetry_path}")
            return
    else:
        print("Exiting program.")
        return

    # End message
    print("\nThank you for using Curvetopia!")
    print("Authors: Rohit Raj, Naitik Verma, Ronan Coutinho")

if __name__ == "__main__":
    main()
