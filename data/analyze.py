filename = 'New_Gowalla/input_node_3000_0.txt'

with open(filename, 'r', encoding='utf-8') as f:
    first_line = f.readline()
    columns = first_line.strip().split('\t')
    print(f'该文件有 {len(columns)} 列')