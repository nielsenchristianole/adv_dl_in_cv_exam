
.PHONY: requirements create_venv clean

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = adv_dl_in_cv_exam
PROJECT_DIR = src
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Set up python interpreter environment
create_venv:
	$(PYTHON_INTERPRETER) -m venv venv

## Install Python Dependencies
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip setuptools wheel
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	$(PYTHON_INTERPRETER) -m pip install -e .

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

annotate:
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/image_annotation/annotate.py

annotate-corners:
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/image_annotation/corner_annotate.py

annotate-elo:
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/image_annotation/elo_annotate.py

plot-wikiart-data:
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/plots/wikiart_plots.py


plot-elo-scores:
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/plots/elo_score_plots.py


sample-diff-timeline:
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/plots/diff_sample_timeline.py --sample

plot-diff-timeline:
	$(PYTHON_INTERPRETER) $(PROJECT_DIR)/plots/diff_sample_timeline.py --plot