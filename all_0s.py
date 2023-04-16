import pandas as pd


ids = pd.read_csv("/Users/eli.seiner/Downloads/test2023_path_to_id.csv")
print(ids['Id'].to_list())

df = pd.DataFrame(columns=['id', 'no finding', 'enlarged cardiomediastinum', 'cardiomegaly', 'lung opacity', 'pneumonia', 'pleural effusion', 'pleural other', 'fracture', 'support devices'])
# 57282
count = 0
for id in ids['Id'].to_list():
    count += 1
    df.loc[len(df.index)] = [id, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if count % 1000 == 0:
        print(count)

df.to_csv("siu4.csv", index=False)