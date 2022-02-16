#%%
from tqdm import tqdm
import os
import re
#%%

folder = '../data/repo_miner/commit_diffs'

word_pattern = r'[\w]+|[^\w\s]'
camelcase_pattern = r'(?<=[a-z])(?=[A-Z])'


symbols = '+-*/%=!<>&|^~.,:"\';{}()[]\\#´`?$'

# %%
tokens = {}

file_type_whitelist = set(['.rs', '.js', '.cxx', '.cpp', '.py', '.c', '.cc', '.ts'])
for file in tqdm(os.listdir(folder)):
    # print(file)
    with open(os.path.join(folder, file), 'r', encoding='utf-8') as f:
        export_diff = f.read()
        header, sep, diff = export_diff.partition('diff --git ')
        
        diff = sep + diff

        for file_export in diff.split('diff --git ')[1:]:
            file_header, sep, file_diff = file_export.partition('\n@@') # everything before first listed source code changes
            file_diff = sep + file_diff
            file_name = file_header.partition('\n')[0].partition(' b/')[2]

            if os.path.splitext(file_name)[1] not in file_type_whitelist:
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

                for wtoken in re.findall(word_pattern, line[1:]):
                    if wtoken in symbols:
                        continue
                    if wtoken.isnumeric():
                        continue
                    for stoken in wtoken.split('_'): # snake case
                        for ctoken in re.split(camelcase_pattern, stoken): # camel case
                            if len(ctoken) > 2:
                                token = prefix + ctoken.lower() # make lower case
                                tokens[token] = tokens.get(token, 0) + 1
# %%
tokens = [(k, v) for k,v in sorted(tokens.items(), key=lambda item: -item[1])]

# %%
for k,v in tokens[:100]:
    print(k ,v)
# %%
re.split(camelcase_pattern, 'camelCaseCase')
# %%

# %%

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

    def __call__(self, file):
        tokens = []
        with open(os.path.join(self.folder, file), 'r', encoding='utf-8') as f:
            export_diff = f.read()
            header, sep, diff = export_diff.partition('diff --git ')
            
            diff = sep + diff

            for file_export in diff.split('diff --git ')[1:]:
                file_header, sep, file_diff = file_export.partition('\n@@') # everything before first listed source code changes
                file_diff = sep + file_diff
                file_name = file_header.partition('\n')[0].partition(' b/')[2]
                
                # exclude generated web assembly files
                # if ('tests' in file_name and 'wasm' in file_name) or '.wast.js' in file_name:
                #     continue
                
                # # exclude js test files
                # if 'js/src/tests/' in file_name:
                #     continue

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
                                    # token = prefix + ctoken.lower() # make lower case
                                    token = ctoken.lower() # make lower case

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

            for file_export in diff.split('diff --git ')[1:]:
                file_header, sep, file_diff = file_export.partition('\n@@') # everything before first listed source code changes
                file_diff = sep + file_diff
                file_name = file_header.partition('\n')[0].partition(' b/')[2]

                if os.path.splitext(file_name)[1] not in self.file_type_whitelist:
                    continue

                print(file_export)
# %%
from sklearn.feature_extraction.text import TfidfVectorizer
folder='../data/repo_miner/commit_diffs'
corpus = os.listdir(folder)
with tqdm(total=len(corpus), desc='Tokenize diffs') as pbar:
    tk = DiffTokenizer(folder=folder, pbar=pbar)
    vt = TfidfVectorizer(tokenizer=tk, norm=None)
    X = vt.fit_transform(corpus)
# %%

# %%
X.shape
# %%
vt.get_feature_names_out()
# %%
import numpy as np
import matplotlib.pyplot as plt
m = np.array(X.max(axis=0).todense()).reshape(-1)
plt.boxplot(m)
# %%
# %%# %%
m.reshape(-1)
# %%
m = np.array(X.max(axis=0).todense()).reshape(-1)
f = sorted(zip(vt.get_feature_names_out(), list(m)), key=lambda x: x[1])
# %%
f[:25]
# %%
f[-25:]
# %%
i = vt.vocabulary_['test262error']
# %%
a = X[:,i].argmax()
a
# %%
X[a,i]

# %%
print(tk.get_source_diff(corpus[a]))
# %%
