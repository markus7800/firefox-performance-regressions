
## Prerequisites

- Python
- Java
- gradle
- hg mercurial
- git
- rust-code-analysis-cli


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

```
python -m src.bugbug_download_dbs
```

### Step 3. Create labelings

#### Run labeling scripts
```
python3 -m src.labeling --bugbug --bugbug_szz_issuelist --fixed_defect_szz_issuelist
```

#### Install SZZ Unleashed

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

#### Converting .hg to .git repository for SZZUnleashed

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

Updating authormap
```
hg log -r 0:tip --template "{author}\n" -R data/mozilla-central/ > utils/authors.txt
```


#### Run SZZUnleashed
```
cd data/labeling/bugbug_szz
java -jar "../szz.jar" -i "./issuelist.json" -r "../../mozilla-central-git" -c 1 -d 1 |& tee output.txt
```

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
```

### Step 5. Run experiments