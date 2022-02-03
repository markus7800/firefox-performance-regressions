# fast-export can't handle some unsuspected author name formats
# this tool was used to find such names and manually make a authormap

import re

rev_ok = 606109 # authormap works at least up to this rev id

names = dict()
i = 0

# ok:
# plain emails: "email" thunder@mozilla.com"
# plain usernames: "user" timeless
# format: "<email>"
# format: "user <email>"
# format: "user<email>"
# "<user> email"
# "user email"

# not ok
# unmatched <>
# "<user>"?
# "user <<email>>"

with open('authors.txt') as f:
    for line in f:
        i += 1
        ok = 0
        if i < rev_ok:
            ok = 1

        name = line.strip('\n')
        if (name.count('<') != name.count('>')) or (not re.match(r'.+\s<[^<>].*>', name)) or name.count('<') > 1 or name.count('>') > 1:
            wasok = names.get(name, 0)
            if wasok == 0:
                names[name] = ok

with open(f'authormap.txt') as f:
    for line in f:
        name = line.partition('"="')[0][1:]
        names[name] = 2



print(f'Found {i} revs, {len(names)} names.')

names = {k: v for k,v in sorted(names.items(), key=lambda x: x[0])}

print('\n\nAUTHORMAP NAMES:\n\n')
for name, ok in names.items():
    if ok == 2:
        print('"' + name + '"="' + name + '"')

print('\n\nOK NAMES:\n\n')
for name, ok in names.items():
    if ok == 1:
        print('"' + name + '"="' + name + '"')


print('\n\nNOT OK NAMES:\n\n')

for name, ok in names.items():
    if ok == 0:
        print('"' + name + '"="' + name + '"')