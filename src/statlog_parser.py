import numpy as np
from tqdm import tqdm
import re
from scipy.sparse import dok_matrix

# parse statlog obtained by
# hg log --git --stat
# to store for each file every revision in which it was changed
# this can be used to assess developer experience and file age

def parse_statlog_to_file_revision_history(path, commit_log):
    renames = []
    files_to_revs = {}
    short_revisions = [] # latest revision last, in accordance with commit_log

    def parse_log_entry(log):
        # example log entry:
        # changeset:   9:ba0e5a8f15da
        # user:        markus
        # date:        Thu Sep 23 23:40:03 2021 +0200
        # summary:     edit a b
        #
        #  a.txt |  5 +----
        #  b.txt |  1 -
        #  2 files changed, 1 insertions(+), 5 deletions(-)
        
        # extract revision (short only 12 characters)
        rev_index, _, short_revision = log[0].split()[1].partition(':')
        rev_index = int(rev_index) # starts at 0

        short_revisions.append(short_revision)

        # if commit is not merge, then second to last line containes summary,
        # like '{} files changed, {} insertions(+), {} deletions(-)'
        if log[-2] == '\n':
            return # commits without file changes, e.g. merges with no files (merges have many files in general)

        # move up from second to last line, until we find empty line -> separates comment from changed lines
        i = -3
        while log[i][0] != '\n':
            i -= 1

        # combine lines to string
        changed_file_text = ''.join(log[i:-2])
        
        # changed files listed as ' filename | {number of modifications} +++---'
        # extract filenames
        pattern = ' (.+?)\s+\|\s+[0-9]+' # excludes binary files (BIN instead of number)
        changed_files = re.findall(pattern, changed_file_text)

        for j in range(len(changed_files)):
            curr_name = ''
            # check for renames and track them
            if ' => ' in changed_files[j]:
                prev_name, _, curr_name = changed_files[j].partition(' => ')
                prev_name = prev_name.replace('\\', '/') # windows compatibility
                curr_name = curr_name.replace('\\', '/')
                
                changed_files[j] = curr_name
                renames.append((prev_name, curr_name))
            else:
                curr_name = changed_files[j].replace('\\', '/')

            # for each file store every revision in which it was changed
            if curr_name not in files_to_revs:
                files_to_revs[curr_name] = set()

            files_to_revs[curr_name].add(rev_index)


        # second to last line
        #pattern = '([0-9]+) files changed, ([0-9]+) insertions\(\+\), ([0-9]+) deletions\(-\)'
        #files_changed, insertions, deletions = map(int, re.findall(pattern, log[-2])[0])

    max_rev = commit_log.iloc[-1].id

    with open(path, encoding='utf-8', errors='ignore') as f: 
        with tqdm(total=max_rev, desc='Parse stat log') as pbar:
            log = []
            for line in f:
                if line[0:9] == 'changeset': # new log entry
                    if log:
                        parse_log_entry(log)
                        pbar.update(1)
                    log = []


                log.append(line)

            parse_log_entry(log)
            pbar.update(1)
            
    assert len(short_revisions) == len(commit_log), 'Number of commits do not match!'
    
       
    # TODO: maybe handle renamed files
     
        
    # write to sparse matrix (one hot encoding)
    file_to_index = {}
    files_revs_history = dok_matrix((len(files_to_revs), max_rev+1), dtype=np.int8)
   

    for file_index, (file, rev_ixs) in tqdm(enumerate(files_to_revs.items()),
                                            total=len(files_to_revs), desc='Write to sparse matrix'):

        file_to_index[file] = file_index

        for ix in rev_ixs:
            files_revs_history[file_index, ix] = 1

    files_revs_history = files_revs_history.tocsr()  
    
    print(f'Found {len(files_to_revs)} unique file names in {len(short_revisions)} revisions with {len(renames)} renames.')
    
    index_to_file = np.array(list(file_to_index.keys()))
    
    # latest revisions on the 'right'
    # e.g. if a commit is in row 5 of commit_log
    # then files_revs_history[:,5] encodes all changed files of this commit
    return file_to_index, index_to_file, files_revs_history

# set entries of revisions not in main branch or merges to 0
def filter_files_revs_history_for_main(orig_files_revs_history, commit_log):
    files_revs_history = dok_matrix(orig_files_revs_history.shape, dtype=np.int8)
    file_indices, rev_indices = orig_files_revs_history.nonzero()
    main_rev_ids = np.array(commit_log.loc[commit_log['main'], 'id'])
    mask = np.isin(rev_indices, main_rev_ids)
    file_indices = file_indices[mask]
    rev_indices = rev_indices[mask]

    for file_index, rev_id in tqdm(zip(file_indices, rev_indices), total = len(rev_indices), desc='Filter files_revs_history for main branch and no merges'):
        files_revs_history[file_index, rev_id] = 1

    files_revs_history = files_revs_history.tocsr() 
    return files_revs_history

def get_modified_files(files_revs_history, index_to_file, rev_index):
    return sorted(index_to_file[files_revs_history[:,rev_index].nonzero()[0]])

def get_revisions_for_file_index(files_revs_history, file_index):
    return sorted(files_revs_history[file_index, :].nonzero()[1])

def get_revisions_for_file(files_revs_history, file_to_index, file):
    return get_revisions_for_file_index(files_revs_history, file_to_index[file])


        
   
   