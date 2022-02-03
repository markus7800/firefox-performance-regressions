from tqdm import tqdm
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd


def get_selected_commits():
    path = 'data/bugbug/commits.json'

    def is_wptsync(commit):
        return "wptsync" in commit["author_email"] or any(
            s in commit["desc"] for s in ("wpt-pr:", "wpt-head:", "wpt-type:")
        )

    all_commits = []
    # contains only default branch and no merges
    with open(path, encoding="utf-8") as f:
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