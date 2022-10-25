# Firefox Performance Regression Prediction

This repository contains source code to reproduce the results of the Master's thesis *Machine Learning for Interactive
Performance Prediction* available [here](https://repositum.tuwien.at/handle/20.500.12708/68310).

## Abstract
>Software performance is an important non-functional project requirement.
Products like web browsers or video games have to keep their loading times low and user interaction smooth to stay competitive in the market.
Since exhaustive testing and benchmarking is infeasible for large-scale projects, other tools have to be integrated in the software development process to ensure a performant system.
>
>In this thesis, we study the open source web browser Mozilla Firefox and build a machine learning model to predict which source code changes are prone to cause performance regressions.
Such a model could be employed to give interactive feedback to developers by raising early warnings for suspicious code, or could be used to help code reviewers to focus their attention on code changes which are likely to have caused a detected performance problem.
>
>The key challenge of predicting performance regressions is the difficulty of data labeling, i.e. determining which code change caused a regression in the past.
After evaluating the SZZ algorithm, commonly used in software defect prediction for this task, to be insufficiently accurate, we present a labeling approach based directly on associations of bug-introducing and bug-fixing issues in the bug-tracking-system.
Even though a lot of effort is put into traditional feature engineering, like computing source code complexity metrics, a bag-of-words model performs best and scores a 5.7 times higher precision than random guessing.
The final model outperforms the best model based on the SZZ algorithm three times with a F1-score of 0.1745, precision of 0.2022 and recall of 0.1535.

## Prerequisites

- Python (install dependencies with `requirements.txt`)
- Java
- gradle
- hg mercurial
- git
- rust-code-analysis-cli

## Project Structure
Here is an overview over the project structure.

Note that some folders/files are only generated when following the step to reproduce the results below.

The data and results used in the thesis/paper can be found [here](https://drive.google.com/drive/folders/1R_z0mZklHMD9NwP4kLRRxV87ZoI9b3zS?usp=sharing).

```
root
└── data
    └── bow/
    └── bugbug/
    └── feature_extractor/
    └── labeling/
    └── mozilla-central
    └── mozilla-central-git
    |   bundle.hg
└── experiments
    └── results/
    └── results_FS/
    |   data_utils.py
    |   hyperparam_tuning.py
    |   modeleval_utils.py
    |   plot_utils.py
    |   requirements.txt
    |   Commit_Versus_Buglevel.ipynb
    |   HyperParameters.ipynb
    |   Interpretability.ipynb
    |   ModelEvaluation.ipynb
    |   Performance_Versus_Regression.ipynb
    |   Sampling.ipynb
    |   SZZevaluation.ipynb
└── src
    |   backout_parser.py
    |   bow_tokenizer.py
    |   bugbug_download_dbs.py
    |   feature_extractor.py
    |   labeling.py
    |   repo_miner.py
    |   statlog_parser.py
    |   utils.py
    |   requirements.txt
└── utils
    └── fast-export/
    └── SZZUnleashed/
    |   authormap.txt
    |   authormap_helper.py
```

## Reproducing Results

### Step 1: Cloning the Firefox repository

Following [instructions](https://firefox-source-docs.mozilla.org/contributing/vcs/mercurial_bundles.html) for mercurial bundles.

```
mkdir data
cd data
```
Go to https://hg.cdn.mozilla.net/ and download mozilla-central "zstd(max)", rename it to `bundle.hg` and put it in the data folder.

```
mkdir mozilla-central
cd mozilla-central
hg init
hg unbundle ../bundle.hg
```

Add
```
[paths]
default = https://hg.mozilla.org/mozilla-central/
```
to the hg config with
```
hg config --local --edit
```

Lastly, run
```
hg pull
hg update
```

### Step 2. Downloading bug information for Firefox

We use a script provided by [bugbug](https://github.com/mozilla/bugbug/tree/master/bugbug).
```
python -m src.bugbug_download_dbs
```

### Step 3. Create labelings

#### Run labeling scripts
```
python3 -m src.labeling --bugbug --fixed_defect_szz_issuelist
```

#### Install [SZZ Unleashed](https://github.com/wogscpar/SZZUnleashed)

```
cd utils
git clone -b mozilla https://github.com/markus7800/SZZUnleashed.git
```

```
cd utils/SZZUnleashed/szz
gradle build
gradle fatJar
```

```
cp utils/SZZUnleashed/szz/build/libs/szz_find_bug_introducers-0.1.jar data/labeling/szz.jar
```

#### Converting .hg to .git repository for SZZUnleashed (takes long)

Use [fast-export](https://github.com/frej/fast-export) to convert.
```
cd utils
git clone https://github.com/markus7800/fast-export.git
```

```
mkdir data/mozilla-central-git
cd data/mozilla-central-git
git init
git config core.ignoreCase false
../../utils/fast-export/hg-fast-export.sh -r ../mozilla-central -A ../../utils/authormap.txt --plugin hg_hash_in_message
```


To map invalid authos to git we use an `utils/authormap.txt`.
The authormap was tested up to revision 606109.

To update it get all authors
```
hg log -r 0:tip --template "{author}\n" -R data/mozilla-central/ > utils/authors.txt
```
and use `utils/authormap_helper`.


#### Run SZZUnleashed

```
cd data/labeling/fixed_defect_szz
java -jar "../szz.jar" -i "./issuelist.json" -r "../../mozilla-central-git" -c 1 -d 1 |& tee output.txt
```

#### Run labeling scripts

```
python3 -m src.labeling --bugbug_szz_export --fixed_defect_szz_export
```

### Step 4. Extract features
```
python3 -m src.feature_extractor
python3 -m src.bow_tokenizer
```

### Step 5. Run experiments
```
# Tuning hyperameters, results will be saved at experiments/results
python3- m experiments.hyperparam_tuning --data=? --model=? ...
```

Various experiments in form of python notebooks in `experiments/`.

These notebooks are not documented well, but can be used to reproduce the experiments of the thesis.

- `Commit_Versus_Buglevel.ipynb`: Show positional bias in bug number based labeling on the commit level.
- `HyperParameters.ipynb`: Print best hyperparameters from search output.
- `Interpretability.ipynb`: Create SHAP interpretability plots.
- `ModelEvaluation.ipynb`: Train models on best hyperparameters and create ROC and precision-recall curve plots as well as evaluation in terms of recall, precision, F1, ROC-AUC, and average precision. 
- `Performance_Versus_Regression.ipynb`: Compare a model trained on performance regressions to a model trained on general defects.
- `Sampling.ipynb`: Get information about best sampling method.
- `SZZevaluation.ipynb`: Evaluate the labeling obtained with SZZUnleashed.
