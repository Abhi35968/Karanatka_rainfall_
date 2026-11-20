with open('d:/PROJECT/karnataka_rainfall_cds/Stage7_Publication_Ready_Package/build_publication_figures.py', 'r') as f:
    text = f.read()

idx = text.find('figure8_extreme_rainfall_errors')
start = text.rfind('FIGURE 8', 0, idx)
print(text[start:idx+100])
