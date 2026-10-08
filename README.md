# UChicago Bayesian Machine Learning

Class activities and notebooks for Bayesian Machine Learning at the University of Chicago.

## Contents

- **Bayes Nets/** - Bayesian networks: structure, inference, and learning
- **Bayesian Inference/** - priors and posteriors for a coin, with a beta prior

## Setup

The Bayes net notebook has an "Open in Colab" badge at the top, which runs it in the browser with nothing to install.
To run the notebooks on your own laptop, follow the steps below.

### 1. Install Graphviz

The Bayes net notebook draws its networks with the Graphviz program, which pip cannot install.

```bash
# macOS
brew install graphviz

# Ubuntu / Debian
sudo apt install graphviz
```

On Windows, download the installer from https://graphviz.org/download/ and tick "Add Graphviz to the system PATH" during the install.
Open a new terminal afterwards so it sees the change.

### 2. Create the virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate it

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 4. Install requirements

```bash
pip install -r requirements.txt
```

### 5. Launch JupyterLab

```bash
jupyter lab
```
