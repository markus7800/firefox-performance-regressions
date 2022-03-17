
prefixes = ['sum', 'mean', 'max', 'min']

personal = {
    'developer_age': 'Developer Seniority',
    'developer_experience': 'Developer Experience',
    'recent_developer_experience': 'Recent Developer Experience',
    'backouts_developer': '# Backouts from Developer',
    'recent_backouts_developer': '# Recent Backouts from Developer'
}

personal = {
    **personal,
    **{
    prefix + '_' + f: f'{d} ({prefix})' for prefix in prefixes for f, d in [
        ('developer_experience_directory', 'Developer Experience in Directory'),
        ('developer_experience_subsystem', 'Developer Experience in Subsystem'),
        ('recent_developer_experience_subsystem', 'Recent Developer Experience in Subsystem'),
        ('recent_developer_experience_directory', 'Recent Developer Experience in Directory')
    ]}
}

diff = {
    'lines_added': '# Lines Added',
    'lines_deleted': '# Lines Deleted',
    'lines_modified': '# Lines Modified',
    'number_of_modified_files': '# Files Modified',
    'number_of_subsystems': '# Subsytems',
    'number_of_directories': '# Directories',
    'entropy_lines_modified': 'Code Entropy',
    'comment_length': 'Comment Length',
    'number_of_commits': '# Commits',
}

diff = {
    **diff,
    **{
    prefix + '_' + f: f'{d} ({prefix})' for prefix in prefixes for f,d in [
        ('backouts_subsystem', '# Backouts Subsystem'),
        ('backouts_directory', '# Backouts Directory'),
        ('recent_backouts_subsystem', '# Recent Backouts Subsystem'),
        ('recent_backouts_directory','# Recent Backouts Directory'),
        ('file_changes', '# Historic File Changes'),
        ('file_ages', 'File Age'),
        ('file_commits_since_last_change', '# Commits Since Last Modified'),
        ('file_number_of_developers', '# Developers')
    ]}
}



complexity = [
    ('nargs_sum', '# Arguments'),
    ('nexits_sum', '# Exits'),
    ('cognitive_sum', 'Cognitive Complexity'),
    ('cyclomatic_sum', 'Cyclomatic Complexity'),
    ('cyclomatic_average', 'Cyclomatic Complexity'),
    ('halstead_n1', 'Halstead n1'),
    ('halstead_N1', 'Halstead N1'),
    ('halstead_n2', 'Halstead n2'),
    ('halstead_N2', 'Halstead N2'),
    ('halstead_length', 'Halstead Length'),
    ('halstead_estimated_program_length', 'Halstead Program Length'),
    ('halstead_purity_ratio', 'Halstead Purity Ratio'),
    ('halstead_vocabulary', 'Halstead Vocabulary'),
    ('halstead_volume', 'Halstead Volume'),
    ('halstead_difficulty', 'Halstead Difficulty',),
    ('halstead_level', 'Halstead Level'),
    ('halstead_effort', 'Halstead Effort'),
    ('halstead_time', 'Halstead Time'),
    ('halstead_bugs', 'Halstead Bugs'),
    ('loc_sloc', '# Source Lines'),
    ('loc_ploc', '# Physical Lines'),
    ('loc_lloc', '# Logical Lines'),
    ('loc_cloc', '# Comment Lines'),
    ('loc_blank', '# Blank Lines'),
    ('nom_functions', '# Functions'),
    ('nom_closures', '# Closures'),
    ('nom_total', '# Methods'),
    ('mi_mi_original', 'Maintainability Index'),
    ('mi_mi_sei', 'Maintainability Index SEI'),
    ('mi_mi_visual_studio', 'Maintainability Index VSC')
]

complexity = {
    **{prefix + '_' + f: f'{d} ({prefix})' for prefix in prefixes for f,d in complexity},
    **{prefix + '_delta_' + f: f'Delta {d} ({prefix})' for prefix in prefixes for f,d in complexity}
}

feature_name_map = {
    **personal,
    **diff,
    **complexity
}