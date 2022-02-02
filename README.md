
## Prerequisites

- Python
- Java
- hg mercurial
- git
- rust-code-analysis-cli


## Cloning the Firefox repository

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

## Downloading bug information for Firefox

```
python -m src.bugbug_download_dbs
```


```
mkdir utils
git clone -b mozilla git@github.com:markus7800/SZZUnleashed.git
git clone https://github.com/frej/fast-export.git
```

## Converting to .git repository for SZZUnleashed

```
mkdir data/mozilla-central-git
cd data/mozilla-central-git
git init
git config core.ignoreCase false
../../utils/fast-export/hg-fast-export.sh -r ../mozilla-central -A ../../utils/authormap.txt 
```

### Updating authormap
```
hg log -r 0:tip --template "{author}\n" -R data/mozilla-central/ > utils/authors.txt
```