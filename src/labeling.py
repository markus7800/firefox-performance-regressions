from tqdm import tqdm
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd


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

def get_regressor_and_fix_bug_ids_by_kind():
    regressor_bug_ids_by_kind = {
        'security': [],
        'memory': [],
        'crash': [],
        'performance': [],
        'power': [],
        'regression': []
        }
    fix_bug_ids_by_kind = {
        'security': [],
        'memory': [],
        'crash': [],
        'performance': [],
        'power': [],
        'regression': []
        }
    fix_bugs = {}

    with open('data/bugbug/bugs.json', encoding="utf-8") as f:
        for line in tqdm(f, desc='Get regressor and fix bug ids'):
            bug = json.loads(line)
            if bug['regressed_by'] and bug["product"] != "Invalid Bugs":
                regressor_bug_ids_by_kind['regression'] = regressor_bug_ids_by_kind['regression'] + bug['regressed_by']
                fix_bug_ids_by_kind['regression'].append(bug['id'])
                fix_bugs[bug['id']] = bug
                
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

    return regressor_bug_ids_by_kind, fix_bug_ids_by_kind, fix_bugs

def get_bugbug_labeling(regressor_bug_ids_by_kind, fix_bugs, selected_commits):

    labeled_commits = []
    for i, commit in selected_commits.iterrows():
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

def get_bugbug_issuelist(fix_bug_ids_by_kind, fix_bugs, selected_commits, hg_to_git):

    issue_list = {}
    date_format = '%Y-%m-%d %X %z'
    for i, commit in selected_commits.iterrows():
        if commit['bug_id'] in fix_bug_ids_by_kind['performance']:
            bug = fix_bugs[commit['bug_id']]
            issue_list[f'issue_{i}'] = {
                'creationdate': pd.to_datetime(bug['creation_time']).strftime(date_format),
                'resolutiondate': pd.to_datetime(bug['last_change_time']).strftime(date_format),
                'hash': hg_to_git[commit['revision']], # convert to git revision
                'commitdate': commit['date'].strftime(date_format) 
            }

    return issue_list

if __name__ == '__main__':
    from src.utils import make_directory
    make_directory('data/labeling')

    regressor_bug_ids_by_kind, fix_bug_ids_by_kind = get_regressor_and_fix_bug_ids_by_kind()
    selected_commits = get_selected_commits()

    bugbug_labeling = get_bugbug_labeling(regressor_bug_ids_by_kind, selected_commits)

    bugbug_labeling.to_csv('data/labeling/bugbug.csv')