# UChicago Bayesian Machine Learning

Class activities and notebooks for Bayesian Machine Learning at the University of Chicago.

## Contents

- **Bayes Nets/** - Bayesian networks: structure, inference, and learning
- **Bayesian Inference/** - priors and posteriors for a coin: five hypotheses and their MAP estimate (`map.ipynb`), then a beta prior (`beta.ipynb`); and a loan officer who decides by expected utility (`credit_loan.ipynb`)
- **Bayesian Changepoint Detection/** - when did the market change? Bayesian online changepoint detection on the daily returns of the Istanbul stock exchange and of an energy fund (`change_point_detect.ipynb`, with its data file `imkb.csv`)
- **Hidden Markov Models/** - calm and turbulent days in the S&P 500: the forward, backward and Viterbi algorithms on the moody friend, then a Gaussian HMM of daily returns, fit by Baum-Welch, with the number of states chosen by BIC (`sp500_hmm.ipynb`; the prices come from Yahoo Finance through `yfinance`)

## Setup

The Bayes net, MAP, credit loan, changepoint and hidden Markov model notebooks have an "Open in Colab" badge at the top, which runs them in the browser with nothing to install.
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
