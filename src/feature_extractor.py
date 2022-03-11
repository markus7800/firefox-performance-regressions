from pyexpat import features
from scipy.stats import entropy
import numpy as np
from tqdm import tqdm
import re
import pandas as pd
import os
from dateutil.relativedelta import relativedelta

from src.statlog_parser import *
from src.backout_parser import *
from src.repo_miner import *
from src.labeling import *
from src.utils import *

def n_added_lines(diff):
    return sum(1 for m in re.finditer('^\+', diff, flags=re.MULTILINE))
def n_deleted_lines(diff):
    return sum(1 for m in re.finditer('^-', diff, flags=re.MULTILINE))
def n_changed_lines(diff):
    return sum(1 for m in re.finditer('^[+-]', diff, flags=re.MULTILINE))

def get_directory_subsystem(file_name):
    file_dirs = file_name.split('/')
    if len(file_dirs) == 1:
            subsystem = 'root'
            directory = 'root'
    else:
        subsystem = file_dirs[0]
        directory = '/'.join(file_dirs[0:-1])
        
    return directory, subsystem

# position in array is 
def index_to_dict(index):
    d = {}
    for i, value in enumerate(index):
        if value not in d:
            d[value] = []

        l = d[value]
        l.append(i)
    
    return d

# if file is deleted no more complexity metrics -> it is not executed anymore
# if file is added -> more complexity
def get_complexity_metrics(path, file_to_index):
    print('Read complexity metrics')
    complexity_metrics = pd.read_csv(path)
    complexity_metrics['file_index'] = [file_to_index[file_name] for file_name in complexity_metrics['file_name']]

    complexity_metrics = complexity_metrics.set_index(['rev_index', 'file_index'])
    complexity_metrics = complexity_metrics.sort_index()
    complexity_metrics = complexity_metrics.drop(['nargs_average', 'nexits_average', 'cognitive_average'], axis=1)
    complexity_metrics = complexity_metrics.drop(['file_name', 'extension'], axis=1)

    n_without_metrics = complexity_metrics['loc_sloc'].isna().sum()
    total_na = complexity_metrics.isna().sum().max()
    print(f'Drop {n_without_metrics} rows without metrics and {total_na - n_without_metrics} rows with missing metrics.')
    
    complexity_metrics = complexity_metrics.dropna() # for now
    
    return complexity_metrics


class FeatureExtractor:
    
    def __init__(self, data_path, file_type_whitelist):
        self.data_path = data_path
        
        commit_log_path = os.path.join(data_path, 'commit_log.csv')
        statlog_path = os.path.join(data_path, 'statlog.txt')
        complexity_metrics_path = os.path.join(data_path, 'complexity_metrics.csv')
        self.diff_dir = os.path.join(data_path, 'commit_diffs/')
        
        commit_log = get_commit_log(commit_log_path)
        
        # collect all files in repository and at which revisions they were modified
        # returns a mapping from filename to index, index to filename and a sparse matrix one-hot encoded files x revision
        file_to_index, index_to_file, files_revs_history = parse_statlog_to_file_revision_history(statlog_path, commit_log)
        
        self.file_to_index = file_to_index
        self.file_type_whitelist = file_type_whitelist
        # filter for main branch and no merges
        self.files_revs_history = filter_files_revs_history_for_main(files_revs_history, commit_log)
        
        complexity_metrics = get_complexity_metrics(complexity_metrics_path, self.file_to_index)
        self.complexity_metrics = complexity_metrics
        
        # to collect which directories and subsystems where modified at which revision
        # get directory and subsystem for each file
        index_to_directory = list(map(lambda x: get_directory_subsystem(x)[0], index_to_file))
        index_to_subsytem = list(map(lambda x: get_directory_subsystem(x)[1], index_to_file))
        
        # collect all (revision) indices for each subsystem / directory
        # e.g. {subsystem1: [1,2,3], sybsystem2: [3,4,5]} means subsystem was modified at revision 1, 2, and 3 etc
        self.directory_to_indices = index_to_dict(index_to_directory)
        self.subsystem_to_indices = index_to_dict(index_to_subsytem)
        
        backouts_df = parse_backouts_from_commit_log(commit_log)
        commit_log = commit_log.join(backouts_df, on='short')

        self.backout_history = np.array(~commit_log['backedoutby'].isna()) # consecutive rev_index important
        
        
        self.commit_log = commit_log

        
     
    # returns boolean vector indicating at which revision directory was modified
    def get_directory_history(self, directory):
        directory_history = np.zeros(self.files_revs_history.shape[1], np.bool8)
        for i in self.files_revs_history[self.directory_to_indices[directory], :].nonzero()[1]:
            directory_history[i] = True
        return directory_history

    # returns boolean vector indicating at which revision subsystem was modified
    def get_subsystem_history(self, subsystem):
        subsystem_history = np.zeros(self.files_revs_history.shape[1], np.bool8)
        for i in self.files_revs_history[self.subsystem_to_indices[subsystem], :].nonzero()[1]:
            subsystem_history[i] = True
        return subsystem_history

    def get_author_history(self, author):
        return np.array(self.commit_log['author'] == author)

    def get_backout_history(self):
        return self.backout_history
        

    def extract_features(self, commits):
        assert len(commits) > 0
        
        # sum over all commits
        lines_added = 0
        lines_deleted = 0
        lines_modified = 0

        subsystems = set() # count number of subsystems (top directory)
        directories = set() # count number of directories
        lines_modified_distribution = []

        modified_files = set()
        # average over all commits
        file_changes = []
        file_ages = [] # in commits
        file_commits_since_last_change = []
        file_number_of_developers = []

        comment_lengths = []

        for commit in commits:
            rev_index = commit['id']

            with open(os.path.join(self.diff_dir, f'{rev_index}.txt'), encoding='utf-8') as f: # read diff
                export_diff = f.read()

                header, sep , diff = ''.join(export_diff).partition('diff --git ')
                comment = re.sub(r'^#.*\n?', '', header, flags=re.MULTILINE) # remove header lines starting with #
                diff = sep + diff # add separator back

                # diff starts with 'diff --git'
                diff = "\n" + diff # hack for separating in next line, maybe replace with regex
                diff_per_file = diff.split("\ndiff --git ")[1:]

                comment_lengths.append(len(comment.split()))  # word count

                # file level metrics + diff metrics
                for i, d in enumerate(diff_per_file):

                    header, _, diff_body = d.partition('\n@@') # everything before first listed source code changes
                    header_lines = header.split('\n')
                    file_name = header_lines[0].partition(' b/')[2]

                    _, extension = os.path.splitext(file_name)

                    # skip irrelevant files
                    if extension not in self.file_type_whitelist:
                        continue

                    if diff_body == '': # binary files
                        continue

                    file_index = self.file_to_index[file_name]

                    a_locs = n_added_lines(diff_body)
                    d_locs = n_deleted_lines(diff_body)

                    lines_added += a_locs
                    lines_deleted += d_locs
                    lines_modified += (a_locs + d_locs)


                    directory, subsystem = get_directory_subsystem(file_name)

                    subsystems.add(subsystem)
                    directories.add(directory)

                    lines_modified_distribution.append(a_locs + d_locs)

                    # revision ids where file was modified up to and excluding current revision (rev_index)
                    file_edit_history = self.files_revs_history[file_index, :rev_index].nonzero()[1]
                    file_changes.append(len(file_edit_history))

                    if len(file_edit_history) == 0:   
                        number_of_developers = 0
                        age = rev_index - 0
                        commits_since_last_change = 0


                    else:
                        first_edit = file_edit_history.min()
                        last_edit = file_edit_history.max()
                        file_authors = self.commit_log.iloc[file_edit_history].author.unique()
                        number_of_developers = len(file_authors)
                        age = rev_index - first_edit
                        commits_since_last_change = rev_index - last_edit


                    file_ages.append(age)
                    file_commits_since_last_change.append(commits_since_last_change)

                    file_number_of_developers.append(number_of_developers)

                f.close()


        
        number_of_modified_files = len(modified_files)
        avg_comment_length = np.mean(comment_lengths)

        # developer, subsystem and directory metrics based on first commit
        # e.g. experience calculated from history made before commits
        first_commit = commits[0]
        author = first_commit['author'] # all commits by same author
        revision = first_commit['revision']
        rev_index = first_commit['id']
        revisions = ','.join(map(lambda commit: commit['revision'], commits))
        rev_indices = ','.join(map(lambda commit: str(commit['id']), commits))

        recent_date = first_commit['date']- relativedelta(months=3)
        recent_revs = (self.commit_log['date'] > recent_date)
           

        # developer metrics

        # count commits by author in project
        author_history = self.get_author_history(author)
        
        developer_age = rev_index - author_history.argmax() # argmax ~ findfirst True for boolean np.array
        developer_experience = author_history[:rev_index].sum()
        recent_developer_experience = (recent_revs & author_history)[:rev_index].sum()

        
        backouts_developer = (author_history & self.backout_history)[:rev_index].sum()
        recent_backouts_developer = (recent_revs & author_history & self.backout_history)[:rev_index].sum()
        
        
        # subsystem metrics

        developer_experience_subsystem = []
        recent_developer_experience_subsystem = []

        backouts_subsystem = []
        recent_backouts_subsystem = []
        
        for subsystem in subsystems:
            sh = self.get_subsystem_history(subsystem)
            
            h = author_history & sh
            developer_experience_subsystem.append(h[:rev_index].sum())
            recent_developer_experience_subsystem.append((recent_revs & h)[:rev_index].sum())
            
            h = self.backout_history & sh
            backouts_subsystem.append(h[:rev_index].sum())
            recent_backouts_subsystem.append((recent_revs & h)[:rev_index].sum())

                                                  
        #print('subsystems', subsystems, developer_experience_subsystem, backouts_subsystem, regressions_subsystem)
        

        # directory metrics

        developer_experience_directory = []
        recent_developer_experience_directory = []
        
        backouts_directory = [] 
        recent_backouts_directory = [] 

        for directory in directories:
            dh = self.get_directory_history(directory)
            
            h = author_history & dh
            developer_experience_directory.append(h[:rev_index].sum())
            recent_developer_experience_directory.append((recent_revs & h)[:rev_index].sum())
            
            h = self.backout_history & dh
            backouts_directory.append(h[:rev_index].sum())
            recent_backouts_directory.append((recent_revs & h)[:rev_index].sum())
                                                  
        #print('directories', directories, developer_experience_directory, backouts_directory, regressions_directory)


        features = {
            'first_revision': revision,
            'first_id': rev_index,
            'revisions': revisions,
            'ids': rev_indices,

            'number_of_commits': len(commits),

            'lines_added': lines_added,
            'lines_deleted': lines_deleted,
            'lines_modified': lines_modified,

            'number_of_modified_files': number_of_modified_files,
            'number_of_subsystems': len(subsystems),
            'number_of_directories': len(directories),
            'entropy_lines_modified': entropy(lines_modified_distribution, base=2),

            'comment_length': avg_comment_length,

            'developer_age': developer_age,
            'developer_experience': developer_experience,
            'recent_developer_experience': recent_developer_experience,
            
            'backouts_developer': backouts_developer,
            'recent_backouts_developer': recent_backouts_developer
        }
        
        
        
        for agg, prefix in [(np.mean, 'mean_'), (np.sum, 'sum_'), (np.min, 'min_'), (np.max, 'max_')]:
            
            features[prefix + 'developer_experience_subsystem'] = agg(developer_experience_subsystem) if len(developer_experience_subsystem) > 0 else np.nan
            features[prefix + 'recent_developer_experience_subsystem'] = agg(recent_developer_experience_subsystem) if len(developer_experience_subsystem) > 0 else np.nan
            features[prefix + 'backouts_subsystem'] = agg(backouts_subsystem) if len(backouts_subsystem) > 0 else np.nan
            features[prefix + 'recent_backouts_subsystem'] = agg(recent_backouts_subsystem) if len(backouts_subsystem) > 0 else np.nan
            
            features[prefix + 'developer_experience_directory'] = agg(developer_experience_directory) if len(developer_experience_directory) > 0 else 0.
            features[prefix + 'recent_developer_experience_directory'] = agg(recent_developer_experience_directory) if len(developer_experience_directory) > 0 else np.nan
            features[prefix + 'backouts_directory'] = agg(backouts_directory) if len(backouts_directory) > 0 else np.nan
            features[prefix + 'recent_backouts_directory'] = agg(recent_backouts_directory) if len(backouts_directory) > 0 else np.nan
                
            features[prefix + 'file_changes'] = agg(file_changes) if len(file_changes) > 0 else np.nan
            features[prefix + 'file_ages'] = agg(file_ages) if len(file_ages) > 0 else np.nan
            features[prefix + 'file_commits_since_last_change'] = agg(file_commits_since_last_change) if len(file_commits_since_last_change) > 0 else np.nan
            features[prefix + 'file_number_of_developers'] = agg(file_number_of_developers) if len(file_number_of_developers) > 0 else np.nan
            
            

        rev_indexes = list(map(lambda commit: commit['id'], commits))
        ixs = [rev_index for rev_index in rev_indexes if rev_index in self.complexity_metrics.index]
        if len(ixs) > 0:
            # compute aggregate complexity of files touched in commits
            commit_complexity_metrics = self.complexity_metrics.loc[ixs,:] # aggregate over all commits

            features = {**features, **(commit_complexity_metrics.sum().add_prefix('sum_'))}
            features = {**features, **(commit_complexity_metrics.mean().add_prefix('mean_'))}
            features = {**features, **(commit_complexity_metrics.max().add_prefix('max_'))}
            features = {**features, **(commit_complexity_metrics.min().add_prefix('min_'))}
        
            # compute complexity delta caused by commits

            first_rev_id = commits[0]['id']
            last_rev_id = commits[-1]['id']
            file_indices = list({file_index for _, file_index in self.complexity_metrics.loc[ixs].index}) # unique

            # guaranteed to have all files
            after_metrics = self.complexity_metrics.loc[pd.IndexSlice[(first_rev_id-1):last_rev_id, file_indices], :]
            after_metrics = after_metrics.groupby('file_index').agg(lambda x: x.iloc[-1]) # select latest metrics

            try:
                # may have files missing because they were added in commits
                before_metrics = self.complexity_metrics.loc[pd.IndexSlice[:(first_rev_id-1), file_indices], :]
                before_metrics = before_metrics.groupby('file_index').agg(lambda x: x.iloc[-1]) # select latest metrics
            except KeyError:
                # every file touched in commits is a new file -> no before metrics
                before_metrics = pd.DataFrame()

            # if before does not have file set all before complexity metrics to 0
            # -> we do not substract anything from after metrics
            delta_metrics = after_metrics.sub(before_metrics, fill_value=0)
            features = {**features, **(delta_metrics.sum().add_prefix('sum_delta_'))}
            features = {**features, **(delta_metrics.mean().add_prefix('mean_delta_'))}
            features = {**features, **(delta_metrics.max().add_prefix('max_delta_'))}
            features = {**features, **(delta_metrics.min().add_prefix('min_delta_'))}

        return features
    
    def extract_features_for_commits(self, grouped_commits):
        features = []
        for commit_group in tqdm(grouped_commits, desc='Extracting features'):
            features.append(self.extract_features(commit_group))
                
        features_df = pd.DataFrame(features)
        return features_df

if __name__ == '__main__':
    make_directory('data/feature_extractor')

    file_type_whitelist = set(['.rs', '.js', '.cxx', '.cpp', '.py', '.c', '.cc', '.ts'])
    feature_extractor = FeatureExtractor('data/repo_miner', file_type_whitelist)

    selected_commits = get_selected_commits()
    
    commits = feature_extractor.commit_log.join(selected_commits[['bug_id']], how='inner')

    if len(commits) == len(selected_commits):
        print(f'Found all {len(selected_commits)} selected_commits in commit_log.')
    else:
        print('Did not find all selected_commits in commit_log. data/bugbug/commits.json and the local repository are not synchronised.')

    assert commits['id'].is_monotonic_increasing, 'Commits are not sorted.'


    # Commit level:

    # create groups of single commits since FeatureExtractor handles only group of commits in form of DataFrame
    grouped_commits = [[dict(row)] for i, row in commits.iterrows()]
    features = feature_extractor.extract_features_for_commits(grouped_commits)

    features = features.rename({'first_revision': 'revision', 'first_id': 'id'}, axis=1)
    features = features.drop(['revisions', 'ids', 'number_of_commits'], axis=1)

    features['index'] = features['revision']
    features.set_index('index', inplace=True)
    features = features.convert_dtypes()
    
    
    features.to_csv('data/feature_extractor/features_commitlevel.csv', index=False)

    # Bug level:

    grouped_commits = group_commits_by_bugid_and_author(commits)
    features = feature_extractor.extract_features_for_commits(grouped_commits)

    assert features['first_id'].is_monotonic_increasing, 'Commits are not sorted.'
    features['index'] = features['first_revision']
    features.set_index('index', inplace=True)
    features = features.convert_dtypes()

    features.to_csv('data/feature_extractor/features_buglevel.csv', index=False)