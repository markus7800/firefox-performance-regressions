import numpy as np
import pandas as pd
import os
import os.path
from tqdm import tqdm
import subprocess
import shutil

from joblib import Parallel, delayed

from src.statlog_parser import *
from src.labeling import *
from src.utils import *
        
# export raw content from file with file_index at revision with rev_index
def cat_file(file_history_dir, repo_path, index_to_file, file_index, rev_index):
        file_name = index_to_file[file_index]
        _, extension = os.path.splitext(file_name)

        write_to = os.path.join(file_history_dir, f'{rev_index}_{file_index}{extension}')
        if os.path.isfile(write_to):
            return # already exists, skip

        file_txt = subprocess.run(f'hg cat -r {rev_index} {file_name} --cwd {repo_path}',
                       capture_output=True,
                       shell=True).stdout
        
        with open(write_to, 'w', encoding='utf-8') as f:
            s = file_txt.decode('utf-8', 'ignore')
            f.write(s)

# export diff for revision with rev_index         
def diff_commit(diff_dir, repo_path, rev_index):
        
    write_to = os.path.join(diff_dir, f'{rev_index}.txt')
    if os.path.isfile(write_to):
        return # already exists, skip

    diff = subprocess.run(f'hg export --git -r {rev_index} -R {repo_path}',
                      capture_output=True,
                      shell=True).stdout

    with open(write_to, 'w', encoding='utf-8') as f:
        s = diff.decode('utf-8', 'ignore')
        f.write(s)


def get_commit_log(path):
    # template = '{rev}\x01{node}\x01{date|isodate}\x01{author}\x01{desc}\x01False\x02'
    commit_log = pd.read_csv(path,
                   delimiter='\x01',
                   lineterminator='\x02',
                   header=None,
                   names=['id', 'revision', 'date', 'author', 'description', 'main'])
                   
    commit_log['date'] = pd.to_datetime(commit_log['date'],utc=True)
    commit_log['index'] = commit_log['revision']
    commit_log['short'] = commit_log['revision'].apply(lambda x: x[:12])
    
    commit_log = commit_log.set_index('index')
    
    # simple date interpolation
    commit_log.loc[commit_log.date < commit_log.iloc[0].date, 'date'] = None
    for i in np.where(commit_log['date'].isna())[0]:
        d_prev = commit_log.iloc[i-1].date
        d_next = commit_log.iloc[i+1].date
        dt = d_next - d_prev
        rev =  commit_log.iloc[i].revision
        commit_log.at[rev, 'date'] = d_prev + dt/2

    return commit_log
            
class RepoMiner:
    
    def __init__(self, repo_path, output_path):
        assert os.path.exists(os.path.join(repo_path, '.hg')), f'repo does not exist at {repo_path}'
        self.repo_path = repo_path
        self.output_path = output_path
        make_directory(self.output_path)
        
        self.diff_dir = os.path.join(self.output_path, 'commit_diffs')
        self.file_history_dir = os.path.join(self.output_path, 'file_history')
        self.file_metrics_dir = os.path.join(self.output_path, 'file_metrics')
        
    def run_commit_log(self):
        print('Running commit log ...')
        template = '{rev}\x01{node}\x01{date|isodate}\x01{author}\x01{desc}\x01False\x02'
        out_path = os.path.join(self.output_path, 'commit_log.csv')
        r1 = subprocess.run(f'hg log -r 0:tip -R {self.repo_path} --template "{template}"',
                       capture_output=True,
                       shell=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(r1.stdout.decode('utf-8', 'ignore'))
            
        
        # now only get revisions for main branch and no merges
        # a second command is faster than including --no-merges --branch default above
        template = '{node}\n'
        r2 = subprocess.run(f'hg log -r 0:tip -R {self.repo_path} --no-merges --branch default --template "{template}"',
                       capture_output=True,
                       shell=True)
        
        commit_log = get_commit_log(out_path)[['id', 'revision', 'date', 'author', 'description', 'main']]
        commit_log.loc[np.array(r2.stdout.decode('utf-8', 'ignore').splitlines()), 'main'] = True    
        commit_log.to_csv(out_path, sep='\x01', line_terminator='\x02', encoding='utf-8', index=False, header=False)
        
        
    def read_commit_log(self):
        print('Reading commit_log ...')
        path = os.path.join(self.output_path, 'commit_log.csv')
        self.commit_log = get_commit_log(path)

        return self.commit_log

    def run_statlog(self):
        print('Running statlog ...')
        out_path = os.path.join(self.output_path, 'statlog.txt')
        subprocess.run(f'hg log -r 0:tip -R {self.repo_path} --git --stat > {out_path}',
                   shell=True)
        
    def update_statlog(self, commit_log):
        print('Updating statlog ...')
        path = os.path.join(self.output_path, 'statlog.txt')
        if not os.path.isfile(path):
            print(f'No statlog to update at {path} -> get full stalog')
            self.run_statlog()
            return
        
        curr_rev = 0
        with open(path, encoding='utf-8') as f:
            for line in f:
                if line[0:9] == 'changeset':
                    rev_index, _, short_revision = line.split()[1].partition(':')
                    curr_rev = int(rev_index)
                    
        if  curr_rev+1 >= len(commit_log): # should only be equality
            print('Already complete statlog.')
            return

        print(f'Log stat since rev {curr_rev+1}.')

        r = subprocess.run(f'hg log -r {curr_rev+1}:tip -R {self.repo_path} --git --stat',
                       capture_output=True,
                       shell=True)
                    
        with open(path, 'a', encoding='utf-8') as f:
            f.write(r.stdout.decode('utf-8', 'ignore'))
            
    
    def read_statlog(self):
        print('Reading statlog ...')
        statlog_path = os.path.join(self.output_path, 'statlog.txt')
        file_to_index, index_to_file, files_revs_history = parse_statlog_to_file_revision_history(statlog_path, self.commit_log)
        self.file_to_index = file_to_index
        self.index_to_file = index_to_file
        self.files_revs_history = files_revs_history
        return file_to_index
    
    # def read_labeling(self, name='multilabeling.csv'):
    #     print('Reading labeling ...')
    #     return get_labeling(os.path.join(self.output_path, name))
    
    
    def filter_for_existing_diff_files(self, commit_rev_ids_set):
        to_delete_set = set()
        for file in tqdm(os.listdir(self.diff_dir), desc='filter for existing files'):
            name, extension = os.path.splitext(file)
            rev_index = int(name)
            if rev_index not in commit_rev_ids_set:
                to_delete_set.add(file)

            commit_rev_ids_set.discard(rev_index)

        return commit_rev_ids_set, to_delete_set
            
    def run_diff_commits(self, commit_rev_ids_set, n_jobs=0, delete=False):
        print('Running diff_commits ... ')
        make_directory(self.diff_dir)
        print(f'{len(commit_rev_ids_set)} revs to diff.')
        filtered_commit_rev_ids_set, to_delete_set = self.filter_for_existing_diff_files(commit_rev_ids_set.copy())
        print(f'{len(filtered_commit_rev_ids_set)} revs to diff after filtering for existing files.')
        print(f'{len(to_delete_set)} revs can be deleted.')
        
        if n_jobs > 0:
            Parallel(n_jobs=n_jobs)(delayed(diff_commit)(self.diff_dir, self.repo_path, rev_id) for rev_id in tqdm(filtered_commit_rev_ids_set, desc='Extract diffs'))
            if delete:
                for file in tqdm(to_delete_set, desc='deleting files'):
                    os.remove(os.path.join(self.diff_dir, file))

    
    # construct (file_index, rev_index)
    def get_cat_indexes(self, filtered_file_index_set, commit_rev_ids_set):
        rows, cols = self.files_revs_history.nonzero()
        cat_indexes = []
        for file_index, rev_index in tqdm(zip(rows, cols), desc='Finding (file,version) to cat'):
            if file_index not in filtered_file_index_set:
                continue
            if rev_index not in commit_rev_ids_set:
                continue
            cat_indexes.append((file_index, rev_index))
            
        return set(cat_indexes)
    
    # For each file and revision, find 'before' revision where file was last edited (for delta metrics)
    def get_before_revs(self, commit_rev_ids_set, cat_indexes):
        before_revs = []
        for file_index, rev_index in tqdm(cat_indexes, desc='Finding "before" revisions'):
            nz = self.files_revs_history[file_index, :rev_index].nonzero()[1]
            if len(nz) > 0:
                before_revs.append((file_index, nz.max()))
                
        before_revs = set(before_revs)
        before_revs = {
            (file_index, rev_index) for file_index, rev_index in before_revs
            if rev_index not in commit_rev_ids_set
        }
        return before_revs
            
    
    def filter_for_existing_files(self, cat_indexes, before_revs, append):
        if append and os.path.isfile(os.path.join(self.output_path, 'complexity_metrics.csv')):
            prev_complexity_metrics_df = self.read_complexity_metrics()
            for (i, row) in tqdm(prev_complexity_metrics_df.iterrows(), desc='filter for existing files from complexity metrics df'):
                rev_index, file_name = i

                cat_indexes.discard((self.file_to_index[file_name], rev_index))
                before_revs.discard((self.file_to_index[file_name], rev_index))
        
        for file in tqdm(os.listdir(self.file_history_dir), desc='filter for existing files from file history dir'):
            name, extension = os.path.splitext(file)
            rev_index, _, file_index = name.partition('_')
            rev_index = int(rev_index)
            file_index = int(file_index)
            
            cat_indexes.discard((file_index, rev_index))
            before_revs.discard((file_index, rev_index))
    
    
    def run_cat_files(self, file_type_whitelist, commit_rev_ids_set, n_jobs=0, append=True):
        print('Running cat_files ...')
        self.file_type_whitelist = file_type_whitelist
        
        
        if not hasattr(self, 'commit_log'):
            self.read_commit_log()
            
        if not hasattr(self, 'file_to_index'):
            self.read_statlog()
                
        make_directory(self.file_history_dir)
        
        filtered_file_index = np.array([v for k,v in self.file_to_index.items() if os.path.splitext(k)[1] in file_type_whitelist])
        filtered_file_index_set = set(filtered_file_index)
        
        
        cat_indexes = self.get_cat_indexes(filtered_file_index_set, commit_rev_ids_set)
        n_files = len({file_index for file_index, rev_index in cat_indexes})
        print(f'Found {len(cat_indexes)} files to cat ({n_files} unique files).')
        
        before_revs = self.get_before_revs(commit_rev_ids_set, cat_indexes)
        print(f'Found {len(before_revs)} before revs.')
        
        self.filter_for_existing_files(cat_indexes, before_revs, append)
        print(f'After filtering for existing files: {len(cat_indexes)} files to cat.')
        print(f'After filtering for existing files: {len(before_revs)} before revs.')
        
        
        if n_jobs > 0:
            res = Parallel(n_jobs=4)(delayed(cat_file)(self.file_history_dir, self.repo_path, self.index_to_file, file_index, rev_index) for file_index, rev_index in tqdm(list(cat_indexes), desc='cat indexes'))

            res = Parallel(n_jobs=4)(delayed(cat_file)(self.file_history_dir, self.repo_path, self.index_to_file, file_index, rev_index) for file_index, rev_index in tqdm(list(before_revs), desc='before revs'))
           
    
    def read_complexity_metrics(self):
        complexity_metrics = pd.read_csv(os.path.join(self.output_path, 'complexity_metrics.csv'))
        complexity_metrics = complexity_metrics.set_index(['rev_index', 'file_name'])
        complexity_metrics = complexity_metrics.sort_index()
        return complexity_metrics
            
    def remove_redundant_from_complexity_metrics(self, file_type_whitelist, commit_rev_ids_set):
        filtered_file_index = np.array([v for k,v in self.file_to_index.items() if os.path.splitext(k)[1] in file_type_whitelist])
        filtered_file_index_set = set(filtered_file_index)
        
        cat_indexes = self.get_cat_indexes(filtered_file_index_set, commit_rev_ids_set)
        n_files = len({file_index for file_index, rev_index in cat_indexes})
        print(f'Found {len(cat_indexes)} files to cat ({n_files} unique files).')
        
        before_revs = self.get_before_revs(commit_rev_ids_set, cat_indexes)
        print(f'Found {len(before_revs)} before revs.')
        print(f'Total {len(cat_indexes) + len(before_revs)}.')

        indexes = cat_indexes.union(before_revs)

        prev_complexity_metrics_df = self.read_complexity_metrics()

        df_index = {(rev_index, self.index_to_file[file_index]) for (file_index, rev_index) in indexes}
        df_index = list(df_index.intersection(set(prev_complexity_metrics_df.index)))


        new_complexity_metrics_df = prev_complexity_metrics_df.loc[df_index, :]
        new_complexity_metrics_df = new_complexity_metrics_df.sort_index()
        
        print(f'Old size {len(prev_complexity_metrics_df)}, new size: {len(new_complexity_metrics_df)}, diff: {len(prev_complexity_metrics_df) - len(new_complexity_metrics_df)}')
        print(len(cat_indexes) + len(before_revs) - len(new_complexity_metrics_df), 'left to compute.')
        new_complexity_metrics_df.to_csv(os.path.join(self.output_path, 'complexity_metrics.csv'), index=True)


    def write_complexity_metrics(self, append=True): 
        print('Write complexity metrics ...')

        complexity_metrics = []
        for file in tqdm(os.listdir(self.file_metrics_dir)):
            _file = file[:-5] # remove .json
            file_name, extension = os.path.splitext(_file)
            rev_index, file_index = file_name.split('_')[-2:]
            rev_index = int(rev_index)
            file_index = int(file_index)

            d = read_data_from_json(os.path.join(self.file_metrics_dir, file))
            flat_metrics = {name + '_' + k: v for name, group in d['metrics'].items() for k,v in group.items()}

            row = {'rev_index': rev_index,
                   'file_name': self.index_to_file[file_index],
                   'extension': extension}

            row = {**row, **flat_metrics}

            complexity_metrics.append(row)
        
        files_with_metrics = set(os.listdir(self.file_metrics_dir))
        for file in tqdm(os.listdir(self.file_history_dir)):
            # correspond to files that are deleted or tool could not parse file -> leave metrics empty
            if (file + '.json') not in files_with_metrics:
                file_name, extension = os.path.splitext(file)
                rev_index, file_index = file_name.split('_')[-2:]
                rev_index = int(rev_index)
                file_index = int(file_index)
                
                row = {'rev_index': rev_index,
                    'file_name': self.index_to_file[file_index],
                    'extension': extension}
                    
                complexity_metrics.append(row)
    
        complexity_metrics_df = pd.DataFrame(complexity_metrics)
        complexity_metrics_df = complexity_metrics_df.set_index(['rev_index', 'file_name'])

        if append:
            prev_complexity_metrics_df = self.read_complexity_metrics()
            complexity_metrics_df = prev_complexity_metrics_df.append(complexity_metrics_df)
            complexity_metrics_df = complexity_metrics_df[~complexity_metrics_df.index.duplicated(keep='last')]
            
        complexity_metrics_df = complexity_metrics_df.sort_index()
            
        complexity_metrics_df.to_csv(os.path.join(self.output_path, 'complexity_metrics.csv'), index=True)
              
    def compute_complexity_metrics(self, n_jobs=4):
        print('Compute complexity metrics ...')
        make_directory(self.file_metrics_dir)
        
        res = subprocess.run(f'rust-code-analysis-cli -m -p {self.file_history_dir} -O json -o {self.file_metrics_dir} -j {n_jobs}',
            shell=True, capture_output=True)


    def remove_file_history_folder(self):
        if os.path.exists(self.file_history_dir):
            shutil.rmtree(self.file_history_dir)
              
    def remove_file_metrics_folder(self):
        if os.path.exists(self.file_metrics_dir):
            shutil.rmtree(self.file_metrics_dir)
        
        
if __name__ == "__main__":
    repo_miner = RepoMiner(repo_path='data/mozilla-central', output_path='data/repo_miner')

    repo_miner.run_commit_log()
    commit_log = repo_miner.read_commit_log()
    print(f'commit_log with {len(commit_log)} rows.')


    repo_miner.update_statlog(commit_log)
    repo_miner.read_statlog()

    selected_commits = get_selected_commits()
    print(f'Selected {len(selected_commits)} commits')


    commit_rev_ids_set = set(commit_log.join(selected_commits[['bug_id']], how='inner')['id'])
    if len(commit_rev_ids_set) == len(selected_commits):
        print(f'Found all {len(selected_commits)} selected_commits in commit_log.')
    else:
        print('Did not find all selected_commits in commit_log. data/bugbug/commits.json and the local repository are not synchronised.')


    repo_miner.run_diff_commits(commit_rev_ids_set, n_jobs=4, delete=True)

    file_type_whitelist = set(['.rs', '.js', '.cxx', '.cpp', '.py', '.c', '.cc', '.ts'])
    repo_miner.run_cat_files(file_type_whitelist, commit_rev_ids_set, n_jobs=4, append=True)

    repo_miner.compute_complexity_metrics()
    repo_miner.write_complexity_metrics(append=True)

    repo_miner.remove_file_history_folder()
    repo_miner.remove_file_metrics_folder()

    # repo_miner.remove_redundant_from_complexity_metrics(file_type_whitelist, commit_rev_ids_set)
        
        
        
              
        
       