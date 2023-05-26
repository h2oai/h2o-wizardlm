# wizardlm
Open-Source Implementation of Wizard LM to turn documents into Q:A pairs for LLM fine-tuning

Based on https://arxiv.org/abs/2304.12244

## Installation
Create a Python3.10 environment and install the dependencies:

```bash
pip install -r requirements.txt
```

## Create WizardLM dataset

Edit the base model, and the number of desired rows in `wizardlm.py`, then run it:

```bash
python wizardlm.py
```

You will end up with a file called `wizard_lm.uuid.json` where uuid is a random string.
