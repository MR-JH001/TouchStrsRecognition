import os
import shutil


FOLDERS = [
    'generator_mixtest',
    'generator_mixtrain',
    'generator_print',
    'generator_written',
]

ANNOTATIONS = [
    'anno.txt',
    'anno_crossEntropy.txt',
    'val_anno.txt',
    'val_anno_crossEntropy.txt'
]

MERGE_SAVE_FOLDER = 'merge_set'


def name_transform(prefix: str, sub: str) -> str:
    return prefix + '_' + sub


if not os.path.exists(MERGE_SAVE_FOLDER):
    os.mkdir(MERGE_SAVE_FOLDER)

# Merge callouts
for annotation in ANNOTATIONS:
    merge_annotation_str = ''
    for folder in FOLDERS:
        with open(os.path.join(folder, annotation), 'r') as lines:
            for line in lines.readlines():
                # print(os.path.join(folder, ), end='')
                merge_annotation_str += name_transform(folder, line)
    with open(os.path.join(MERGE_SAVE_FOLDER, annotation), mode='w') as merge_annotation_file:
        merge_annotation_file.write(merge_annotation_str)
# Merge pictures
for folder in FOLDERS:
    for file_name in os.listdir(folder):
        # txt end does not move
        if file_name[-1] == 't':
            continue
        source = os.path.join(folder, file_name)
        destination = os.path.join(MERGE_SAVE_FOLDER, name_transform(folder, file_name))
        shutil.copyfile(source, destination)
