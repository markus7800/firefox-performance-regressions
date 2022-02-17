from tqdm import tqdm
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
import subprocess
from collections import defaultdict
import argparse
import sys
import copy

def get_selected_commits():
    def is_wptsync(commit):
        return "wptsync" in commit["author_email"] or any(
            s in commit["desc"] for s in ("wpt-pr:", "wpt-head:", "wpt-type:")
        )

    all_commits = []
    # contains only default branch and no merges
    with open('data/bugbug/commits.json', encoding="utf-8") as f:
        for line in tqdm(f, desc='Get selected commits'):
            commit_data = json.loads(line)
            all_commits.append([
                commit_data['node'],
                commit_data['pushdate'],
                commit_data['author'],
                commit_data['desc'],
                commit_data['bug_id'],
                commit_data['ignored'],
                commit_data["backedoutby"] != '',
                len(commit_data['backsout']) > 0,
                is_wptsync(commit_data)
                ])

    all_commits = pd.DataFrame(all_commits, columns=[
        'revision',
        'date',
        'author',
        'description',
        'bug_id',
        'ignored',
        'backedout',
        'backsout',
        'is_wptsync']
        )

    all_commits['index'] = all_commits['revision']
    all_commits.set_index('index', inplace=True)
    all_commits = all_commits.convert_dtypes()
    all_commits['date'] = pd.to_datetime(all_commits['date'])


    MIN_DATE = datetime(2019,7,1)

    MAX_DATE = all_commits['date'].max()
    MAX_DATE = datetime.fromtimestamp(MAX_DATE.timestamp())

    selected_commits = all_commits[
        ~all_commits['bug_id'].isna() &
        ~all_commits['backsout'] &
        ~all_commits['is_wptsync'] &
        ~all_commits['ignored'] &
        (MIN_DATE <= all_commits['date']) &
        (all_commits['date'] <= MAX_DATE - relativedelta(months=3))
        ]
    
    return selected_commits
    
def group_commits_by_bugid_and_author(commits):
    # get consecutive commits with same author and bug id
    assert commits['id'].is_monotonic_increasing, 'Commits are not sorted.'

    bug_id = 0
    author = ''
    cs = []
    grouped_commits = []

    def append_group(cs):
        if len(cs) > 0:
            # cs = pd.DataFrame(cs)
            grouped_commits.append(cs)
            
    for (i, row) in commits.iterrows():
        if bug_id != row['bug_id'] or author != row['author']:
            append_group(cs)
            cs = [dict(row)]
            bug_id = row['bug_id']
            author = row['author']
        else:
            cs.append(dict(row))

    append_group(cs)

    assert sum(len(group) for group in grouped_commits) == len(commits), 'Mismatching number of commits.'
    
    return grouped_commits

#https://wiki.mozilla.org/BMO/UserGuide/BugFields
# adapted from https://github.com/mozilla/bugbug/blob/master/bugbug/models/bugtype.py

KEYWORD_DICT = {
    "sec-critical": "security",
    "sec-high": "security",
    "sec-moderate": "security",
    "sec-low": "security",
    "sec-other": "security",
    "sec-audit": "security",
    "sec-vector": "security",
    "sec-want": "security",
    "csectype-bounds": "security",
    "csectype-disclosure": "security",
    "csectype-dos": "security",
    "csectype-framepoisoning": "security",
    "csectype-intoverflow": "security",
    "csectype-jit": "security",
    "csectype-nullptr": "security",
    "csectype-oom": "security",
    "csectype-other": "security",
    "csectype-priv-escalation": "security",
    "csectype-race": "security",
    "csectype-sop": "security",
    "csectype-spoof": "security",
    "csectype-uaf": "security",
    "csectype-undefined": "security",
    "csectype-uninitialized": "security",
    "csectype-wildptr": "security",
    
    "memory-footprint": "memory",
    "memory-leak": "memory",
    
    "crash": "crash",
    "crashreportid": "crash",
    
    "perf": "performance",
    "topperf": "performance",
    "perf-alert": "performance", # added
    
    "power": "power",
}
REGRESSION_TYPES = sorted(set(KEYWORD_DICT.values()))


def bug_to_types(bug):
    types = set()

    if "[overhead" in bug["whiteboard"].lower():
        types.add("memory")

    if "[power" in bug["whiteboard"].lower():
        types.add("power")

    if any(
        f"[{whiteboard_text}" in bug["whiteboard"].lower()
        for whiteboard_text in ("fxperf", "snappy")
    ):
        types.add("performance")

    if "cf_crash_signature" in bug and bug["cf_crash_signature"] not in ("", "---"):
        types.add("crash")

    return list(
        types.union(
            set(
                KEYWORD_DICT[keyword]
                for keyword in bug["keywords"]
                if keyword in KEYWORD_DICT
            )
        )
    )

def get_bugbug_regressors_and_fixes():
    regressor_bug_ids_by_kind = defaultdict(lambda: [])
    fix_bug_ids_by_kind = defaultdict(lambda: [])
    fixes = {}

    with open('data/bugbug/bugs.json', encoding="utf-8") as f:
        for line in tqdm(f, desc='Get regressors and fixes'):
            bug = json.loads(line)
            if bug['regressed_by'] and bug["product"] != "Invalid Bugs":
                regressor_bug_ids_by_kind['regression'] = regressor_bug_ids_by_kind['regression'] + bug['regressed_by']
                fix_bug_ids_by_kind['regression'].append(bug['id'])
                fixes[bug['id']] = bug
                
                for t in bug_to_types(bug):
                    regressor_bug_ids_by_kind[t] = regressor_bug_ids_by_kind[t] + bug['regressed_by']
                    fix_bug_ids_by_kind[t].append(bug['id'])
                

    regressor_bug_ids_by_kind = {k: set(v) for k,v in regressor_bug_ids_by_kind.items()}
    fix_bug_ids_by_kind = {k: set(v) for k,v in fix_bug_ids_by_kind.items()}
    
    print('Found')
    for k,v in regressor_bug_ids_by_kind.items():
        print(len(v), k)
    print('regressor bug ids,')

    print('\nand\n')
    for k,v in fix_bug_ids_by_kind.items():
        print(len(v), k)
    print('fix bug ids.')

    return regressor_bug_ids_by_kind, fix_bug_ids_by_kind, fixes

def get_bugbug_labeling(regressor_bug_ids_by_kind, selected_commits):

    labeled_commits = []
    for _, commit in selected_commits.iterrows():
        row = {
            'revision': commit['revision'],
            'bug_id': commit['bug_id'],
        }
        
        for bugtype, regressor_ids in regressor_bug_ids_by_kind.items():
            if commit["bug_id"] in regressor_ids:
                row[bugtype] = 1
            else:
                row[bugtype] = 0  

        labeled_commits.append(row)

    labeling = pd.DataFrame(labeled_commits)
    labeling = labeling.convert_dtypes()
    labeling.set_index('revision', inplace=True)
    return labeling

def get_defects_and_fixes():
    fix_bug_ids_by_kind = defaultdict(lambda: [])
    fixes = {}
    with open('data/bugbug/bugs.json', encoding="utf-8") as f:
        for line in tqdm(f, desc='Get defects and fixes'):
            bug = json.loads(line)
            if (bug['product'] !='"Invalid Bugs' and
                bug['resolution'] == 'FIXED' and
                bug['type'] == 'defect'):

                fix_bug_ids_by_kind['regression'].append(bug['id'])
                fixes[bug['id']] = bug
    
                for t in bug_to_types(bug):
                    fix_bug_ids_by_kind[t].append(bug['id'])

    print('Found')
    for k,v in fix_bug_ids_by_kind.items():
        print(len(v), k)
    print('fix bug ids.')

    return fix_bug_ids_by_kind, fixes


def get_hg_git_mapping():
    print('Get hg to git mapping ...', end='\r')
    git_repo = 'data/mozilla-central-git'

    # we stored the hg_hash in the commit message
    template = '%H %s'
    git_out_cmd = subprocess.run(f'git log --pretty="{template}"',
                        capture_output=True,
                        shell=True,
                        cwd=git_repo)
    git_log_str = git_out_cmd.stdout.decode('utf-8', 'ignore')

    hg_to_git = {}
    git_to_hg = {}
    for line in git_log_str.splitlines():
        git_hash, _, hg_hash = line.partition(' ')
        hg_to_git[hg_hash] = git_hash
        git_to_hg[git_hash] = hg_hash
    print('Get hg to git mapping ... Done.')

    return hg_to_git, git_to_hg

def get_issuelist(fix_bug_ids_by_kind, fixes, selected_commits, hg_to_git, target='performance'):

    issue_list = {}
    date_format = '%Y-%m-%d %X %z'
    for i, commit in selected_commits.iterrows():
        if commit['bug_id'] in fix_bug_ids_by_kind[target]:
            bug = fixes[commit['bug_id']]
            issue_list[f'issue_{i}'] = {
                'creationdate': pd.to_datetime(bug['creation_time']).strftime(date_format),
                'resolutiondate': pd.to_datetime(bug['last_change_time']).strftime(date_format),
                'hash': hg_to_git[commit['revision']], # convert to git revision
                'commitdate': commit['date'].strftime(date_format) 
            }

    return issue_list

def get_labeling_from_szz_results(path, selected_commits, git_to_hg, target):
    git_szz_results = read_data_from_json(os.path.join(path, 'results/fix_and_introducers_pairs.json'))

    # convert back to hg hashes
    szz_results = copy.deepcopy(git_szz_results)
    for pair in szz_results:
        pair[0] = git_to_hg[pair[0]]
        pair[1] = git_to_hg[pair[1]]

    fix_introducer_pairs = pd.DataFrame(szz_results, columns=['fix', 'introducer'])

    introducers = set(fix_introducer_pairs['introducer'])

    labeled_commits = []
    for _, commit in selected_commits.iterrows():
        row = {
            'revision': commit['revision'],
            'bug_id': commit['bug_id'],
        }
        
        if commit['revision'] in introducers:
            row[target] = 1
        else:
            row[target] = 0 

        labeled_commits.append(row)

    labeling = pd.DataFrame(labeled_commits)
    labeling = labeling.convert_dtypes()
    labeling.set_index('revision', inplace=True)
    return labeling

def print_labeling_stats(labeling, target):
    p = labeling[target].sum()
    n = (1-labeling[target]).sum()
    print(f'positive labels: {p} {p/(p+n)*100:.2f}%, negative labels: {n} {n/(p+n)*100:.2f}%')

if __name__ == '__main__':
    from src.utils import *
    make_directory('data/labeling')
    
    parser = argparse.ArgumentParser()
    bugbug_description = 'Export labeling based on bugzilla regressed by label.'
    parser.add_argument('--bugbug', action="store_true", dest='bugbug', help=bugbug_description)
    
    bugbug_szz_issuelist_description = 'Export issuelist for SZZUnleashed based on bugzilla regressed by label.'
    parser.add_argument('--bugbug_szz_issuelist', action="store_true", dest='bugbug_szz_issuelist',
        help=bugbug_szz_issuelist_description)

    fixed_defect_szz_issuelist_description = 'Export issuelist for SZZUnleashed based on fixed defects.'
    parser.add_argument('--fixed_defect_szz_issuelist', action="store_true", dest='fixed_defect_szz_issuelist',
        help=fixed_defect_szz_issuelist_description)

    bugbug_szz_export_description = 'Export results of SZZUnleashed based on bugzilla regressed by label.'
    parser.add_argument('--bugbug_szz_export', action="store_true", dest='bugbug_szz_export',
        help=bugbug_szz_export_description)

    fixed_defect_szz_export_description = 'Export results of SZZUnleashed based on fixed defects.'
    parser.add_argument('--fixed_defect_szz_export', action="store_true", dest='fixed_defect_szz_export',
        help=fixed_defect_szz_export_description)

    parser.add_argument('--target', type=str, default='performance', dest='target')

    args = parser.parse_args(sys.argv[1:])
    print(f'\n{args=}\n')
    
    selected_commits = get_selected_commits()

    hg_to_git, git_to_hg = {}, {}
    if any([
        args.bugbug_szz_issuelist,
        args.fixed_defect_szz_issuelist,
        args.bugbug_szz_export,
        args.fixed_defect_szz_export
    ]):
        hg_to_git, git_to_hg = get_hg_git_mapping()

    if args.bugbug or args.bugbug_szz_issuelist:
        regressor_bug_ids_by_kind, fix_bug_ids_by_kind, fixes = get_bugbug_regressors_and_fixes()

        if args.bugbug:
            print(bugbug_description)
            labeling = get_bugbug_labeling(regressor_bug_ids_by_kind, selected_commits)
            print_labeling_stats(labeling, args.target)
            labeling.to_csv('data/labeling/bugbug.csv')

        if args.bugbug_szz_issuelist:
            print(bugbug_szz_issuelist_description)
            issuelist = get_issuelist(fix_bug_ids_by_kind, fixes, selected_commits, hg_to_git, target=args.target)
            print(f'{len(issuelist)=}')
            make_directory('data/labeling/bugbug_szz')
            write_json_to_file(issuelist, 'data/labeling/bugbug_szz/issuelist.json')

    if args.fixed_defect_szz_issuelist:
        print(fixed_defect_szz_issuelist_description)
        fix_bug_ids_by_kind, fixes = get_defects_and_fixes()
        issuelist = get_issuelist(fix_bug_ids_by_kind, fixes, selected_commits, hg_to_git, target=args.target)
        print(f'{len(issuelist)=}')
        make_directory('data/labeling/fixed_defect_szz')
        write_json_to_file(issuelist, 'data/labeling/fixed_defect_szz/issuelist.json')

    if args.bugbug_szz_export:
        print(bugbug_szz_export_description)
        labeling = get_labeling_from_szz_results('data/labeling/bugbug_szz', selected_commits, git_to_hg, args.target)
        print_labeling_stats(labeling, args.target)
        labeling.to_csv('data/labeling/bugbug_szz.csv')
    
    if args.fixed_defect_szz_export:
        print(fixed_defect_szz_export_description)
        labeling = get_labeling_from_szz_results('data/labeling/fixed_defect_szz', selected_commits, git_to_hg, args.target)
        print_labeling_stats(labeling, args.target)
        labeling.to_csv('data/labeling/fixed_defect_szz.csv')