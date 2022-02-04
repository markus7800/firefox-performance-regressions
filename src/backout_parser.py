
# adapted from mozautomation/commitparser

import pandas as pd
from tqdm import tqdm
import re

BACKOUT_KEYWORD = r'^(?:backed out|backout|back out)\b'
BACKOUT_KEYWORD_RE = re.compile(BACKOUT_KEYWORD, re.I)
CHANGESET_KEYWORD = r'(?:\b(?:changeset|revision|change|cset|of)\b)'
CHANGESETS_KEYWORD = r'(?:\b(?:changesets|revisions|changes|csets|of)\b)'
SHORT_NODE = r'([0-9a-f]{12}\b)'
SHORT_NODE_RE = re.compile(SHORT_NODE, re.I)

BACKOUT_SINGLE_RE = re.compile(
    BACKOUT_KEYWORD + r'\s+' +
    CHANGESET_KEYWORD + r'?\s*' +
    r'(?P<node>' + SHORT_NODE + r')',
    re.I
)

BACKOUT_MULTI_SPLIT_RE = re.compile(
    BACKOUT_KEYWORD + r'\s+' +
    r'(?P<count>\d+)\s+' +
    CHANGESETS_KEYWORD,
    re.I
)

BACKOUT_MULTI_ONELINE_RE = re.compile(
    BACKOUT_KEYWORD + r'\s+' +
    CHANGESETS_KEYWORD + r'?\s*' +
    r'(?P<nodes>(?:(?:\s+|and|,)+' + SHORT_NODE + r')+)',
    re.I
)

def is_backout(commit_desc):
    """Returns True if the first line of the commit description appears to
    contain a backout.

    Backout commits should always result in is_backout() returning True,
    and parse_backouts() not returning None.  Malformed backouts may return
    True here and None from parse_backouts()."""
    return BACKOUT_KEYWORD_RE.match(commit_desc) is not None

def parse_backouts(commit_desc, strict=False):
    """Look for backout annotations in a string.

    Returns a nodes where each entry is an iterable of
    changeset identifiers that were backed out, respectively.
    Or return None if no backout info is available.

    Setting `strict` to True will enable stricter validation of the commit
    description (eg. ensuring N commits are provided when given N commits are
    being backed out).
    """
    if not is_backout(commit_desc):
        return None

    lines = commit_desc.splitlines()
    first_line = lines[0]

    # Single backout.
    m = BACKOUT_SINGLE_RE.match(first_line)
    if m:
        return [m.group('node')]

    # Multiple backouts, with nodes listed in commit description.
    m = BACKOUT_MULTI_SPLIT_RE.match(first_line)
    if m:
        expected = int(m.group('count'))
        nodes = []
        for line in lines[1:]:
            single_m = BACKOUT_SINGLE_RE.match(line)
            if single_m:
                nodes.append(single_m.group('node'))
        if strict:
            # The correct number of nodes must be specified.
            if expected != len(nodes):
                return None
        return nodes

    # Multiple backouts, with nodes listed on the first line
    m = BACKOUT_MULTI_ONELINE_RE.match(first_line)
    if m:
        return SHORT_NODE_RE.findall(m.group('nodes'))

    return None

def parse_backouts_from_commit_log(commit_log):
    backouts = {}
    with tqdm(total=len(commit_log), desc='Parse backouts') as pbar:
        for i, row in commit_log.iterrows():
            node = row['revision']
            desc = row['description']
            nodes = parse_backouts(desc)
            if nodes:
                for backedout_node in nodes:
                    backouts[backedout_node] = node
            pbar.update(1)
    backouts_df = pd.DataFrame(backouts.items(), columns=['short', 'backedoutby'])
    backouts_df.set_index('short', inplace=True)
    return backouts_df

#def backouts_df_to_vec(backouts_df, commit_log):
#    commit_log['short'] = commit_log['revision'].apply(lambda x: x[:12])
#    short_id_map = commit_log[['short', 'id', 'revision']]
#    short_id_map.set_index('short', inplace=True)
#    
#    commit_backouts = backouts_df.join(short_id_map, how='inner')
#    
#    rev_ids = commit_backouts['id'] # lose some revisions which are wrongly written in description
#    commit_log['backedout'] = False
#    commit_log.loc[commit_backouts['revision'], 'backedout'] = True
#    
#    # one hot encode
#    backouts_vec = np.zeros(len(commit_log), np.bool8)
#    for i in rev_ids:
#        backouts_vec[i] = True
#    
#    return backouts_vec
#
#def read_backouts(path, commit_log):
#    backouts_df = pd.read_csv(path, index_col=0)
#    return backouts_df
    
    
    