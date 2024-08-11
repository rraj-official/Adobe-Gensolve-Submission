
# Curvetopia

Video Link: https://www.youtube.com/watch?v=fr7JUgmusgs

**Welcome to Curvetopia: A Journey into the World of Curves**

This project involves processing 2D curves from polylines to cubic Bézier curves, with tasks that include regularizing curves, exploring symmetry, and completing incomplete curves.
Team Members:
- Rohit Raj
- Naitik Verma
- Ronan Coutinho

## Setup

### Step 1: Install and Activate Virtual Environment

#### For macOS/Linux:

1. Open your terminal and navigate to the project directory.
2. Run the following commands:

```bash
python3 -m venv env
source env/bin/activate
```

#### For Windows:

1. Open Command Prompt or PowerShell and navigate to the project directory.
2. Run the following commands:

```bash
python -m venv env
.\env\Scripts\activate
```

### Step 2: Install Dependencies

After activating the virtual environment, install the required dependencies using:

```bash
pip install -r dependencies.txt
```

### Step 3: Prepare Input Files

Before running the program, ensure that the necessary input files are placed in the `input` folder:

- **symmetry.png**: A PNG file for checking the line of symmetry.
- **regularize.svg**: An SVG file for regularizing the lines and curves.

**Note:** Sample input files for both tasks have been provided in the `input` folder.

### Step 4: Run the Program

To run the program, execute the following command:

```bash
python main.py
```

**Note:** Close popups to continue the program, as some popups might appear during execution. The generated files shown in the popups will also be saved in the `outputs` folder.

---
