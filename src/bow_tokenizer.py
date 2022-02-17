from tqdm import tqdm
import os
import re

class DiffTokenizer:
    def __init__(self, folder, pbar=None):
        self.folder = folder
        self.word_pattern = r'[\w]+|[^\w\s]'
        self.camelcase_pattern = r'(?<=[a-z])(?=[A-Z])'
        self.symbols = '+-*/%=!<>&|^~.,:"\';{}()[]\\#´`?$'
        self.file_type_whitelist = set(['.rs', '.js', '.cxx', '.cpp', '.py', '.c', '.cc', '.ts'])
        self.pbar = pbar

    def __call__(self, commits):
        tokens = []
        for commit in commits:
            rev_index = commit['id']
            file = f'{rev_index}.txt'
            with open(os.path.join(self.folder, file), 'r', encoding='utf-8') as f:
                export_diff = f.read()
                header, sep, diff = export_diff.partition('diff --git ')
                
                diff = sep + diff

                for file_export in diff.split('diff --git ')[1:]:
                    file_header, sep, file_diff = file_export.partition('\n@@') # everything before first listed source code changes
                    file_diff = sep + file_diff
                    file_name = file_header.partition('\n')[0].partition(' b/')[2]
                    
                    # exclude generated web assembly files
                    if ('tests' in file_name and 'wasm' in file_name) or '.wast.js' in file_name:
                        continue
                    
                    # exclude js test files
                    if 'js/src/tests/' in file_name:
                        continue

                    if os.path.splitext(file_name)[1] not in self.file_type_whitelist:
                        continue

                    for line in file_diff.splitlines():
                        if len(line) == 0:
                            continue
                        if line[:10] == 'diff --git' or line[:3] == '+++' or line[:3] == '---' or line[:2] == '@@':
                            continue

                        prefix = ''
                        if line[0] == '+':
                            prefix = 'added_'
                        elif line[0] == '-':
                            prefix = 'deleted_'
                        else:
                            prefix = 'context_'

                        for wtoken in re.findall(self.word_pattern, line[1:]):
                            if wtoken in self.symbols:
                                continue
                            if wtoken.isnumeric():
                                continue
                            for stoken in wtoken.split('_'): # snake case
                                for ctoken in re.split(self.camelcase_pattern, stoken): # camel case
                                    if len(ctoken) > 2:
                                        token = prefix + ctoken.lower() # make lower case
                                        
                                        tokens.append(token)
        if self.pbar:
            self.pbar.update(1)

        return tokens

    def get_diff(self, file):
        with open(os.path.join(self.folder, file), 'r', encoding='utf-8') as f:
            export_diff = f.read()
            return export_diff

    def get_source_diff(self, file):
        with open(os.path.join(self.folder, file), 'r', encoding='utf-8') as f:
            export_diff = f.read()
            header, sep, diff = export_diff.partition('diff --git ')
            print(header)
            print('\n\n')

            for file_export in diff.split('diff --git ')[1:]:
                file_header, sep, file_diff = file_export.partition('\n@@') # everything before first listed source code changes
                file_diff = sep + file_diff
                file_name = file_header.partition('\n')[0].partition(' b/')[2]

                if os.path.splitext(file_name)[1] not in self.file_type_whitelist:
                    continue

                print(file_export)


import numpy as np
from src.utils import make_directory
from src.labeling import *
from src.repo_miner import get_commit_log
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

if __name__ == '__main__':
    selected_commits = get_selected_commits()

    commit_log = get_commit_log('data/repo_miner/commit_log.csv')

    commits = commit_log.join(selected_commits[['bug_id']], how='inner')

    if len(commits) == len(selected_commits):
        print('Found all selected_commits in commit_log.')
    else:
        print('Did not find all selected_commits in commit_log. data/bugbug/commits.json and the local repository are not synchronised.')

    assert commits['id'].is_monotonic_increasing, 'Commits are not sorted.'


    max_features = 50000
    

    # Commit level:
    grouped_commits = [[dict(row)] for i, row in commits.iterrows()]

    folder='data/repo_miner/commit_diffs'
    with tqdm(total=len(grouped_commits), desc='Tokenize diffs - Commit level') as pbar:
        tk = DiffTokenizer(folder=folder, pbar=pbar)
        # lowercase=False because input are not real docs but commits, disables any preprocessing
        vt = TfidfVectorizer(tokenizer=tk, max_features=max_features, lowercase=False) 
        X = vt.fit_transform(grouped_commits)
        with open('data/bow/bow_commitlevel.pickle', 'wb') as f:
            pickle.dump((X, vt.get_feature_names_out()), f)
        
        revisions = pd.DataFrame([(commits[0]['revision'], commits[0]['id']) for commits in grouped_commits], columns=['revision', 'id'])
        revisions.to_csv('data/bow/revisions_commitlevel.csv', index=False)

    
    # Bug level:
    grouped_commits = group_commits_by_bugid_and_author(commits)
    with tqdm(total=len(grouped_commits), desc='Tokenize diffs - Bug level') as pbar:
        tk = DiffTokenizer(folder=folder, pbar=pbar)
        # lowercase=False because input are not real docs but commits, disables any preprocessing
        vt = TfidfVectorizer(tokenizer=tk, max_features=max_features, lowercase=False) 
        X = vt.fit_transform(grouped_commits)
        with open('data/bow/bow_buglevel.pickle', 'wb') as f:
            pickle.dump((X, vt.get_feature_names_out()), f)
        
        revisions = pd.DataFrame([(commits[0]['revision'], commits[0]['id'])  for commits in grouped_commits], columns=['first_revision', 'first_id'])
        revisions.to_csv('data/bow/revisions_buglevel.csv', index=False)