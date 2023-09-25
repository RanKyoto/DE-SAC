import os

def mkdir(path:str):
    path = path.strip()
    path = path.rstrip("\\")

    isExists=os.path.exists(path)

    if not isExists:
        os.makedirs(path) 
        print('path:' + path + ' done.')
    else:
        print('path:' + path + ' exist.')
    return True

def process_bar(percent, start_str='', end_str='', total_length=30):
    bar = ''.join(['■'] * int(percent * total_length)) + ''
    bar = '\r' + start_str + bar.ljust(total_length) + ' {:0>4.1f}%|'.format(percent*100) + end_str
    print(bar, end='', flush=True)