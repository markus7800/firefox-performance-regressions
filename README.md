
## Prerequisites

- hg mercurial
- git


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
python -m scripts.bugbug_download_dbs
```