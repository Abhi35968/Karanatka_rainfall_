with open('d:/PROJECT/karnataka_rainfall_cds/Stage9_2_Final_Author_Review/Karnataka_Rainfall_Research_Paper_Final.md', 'r', encoding='utf-8') as f:
    txt = f.read()

import re
paths = re.findall(r'[a-zA-Z]:[\\/][^\s\)\"\'>]+', txt)
print('Paths found in MD:', len(paths), paths)

