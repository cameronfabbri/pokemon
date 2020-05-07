import os

data_dir = os.path.join('data','pokemon','bw_ani')
# find . ! -empty -type f -exec md5sum {} + | sort | uniq -w32 -dD
with open(os.path.join(data_dir, 'duplicates.txt'), 'r') as f:

    data = {}
    for line in f:
        line = line.rstrip().split()
        hash_num = line[0]
        file = line[-1].replace('./','')
        file = os.path.join(data_dir, file)
        data[hash_num] = file

for h, file in data.items():
    filename = os.path.basename(file)
    new_name = os.path.join(data_dir, 'temp', filename)
    print(file, '-->', new_name)
    os.rename(file, new_name)
