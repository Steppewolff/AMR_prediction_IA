import numpy as np
import pandas as pd

aspire = pd.read_csv('ASPIRE.csv')
colimero = pd.read_csv('COLIMERO.csv')
gemara = pd.read_csv('GEMARA.csv')
scores = pd.read_csv('SCORES.csv')
tables2 = pd.read_csv('TABLES2.csv')

files = [aspire, colimero, gemara, scores, tables2]

df_columns = ["ID","AZT_prof","IMI_prof","FEP_prof","PA0004","PA0005","PA0424","PA0425","PA0426","PA0427","PA0807","PA0869","PA0958","PA1179","PA1180","PA1777","PA1798","PA1799","PA2018","PA2019","PA2020","PA2023","PA2272","PA2491","PA2492","PA2493","PA2494","PA2495","PA3047","PA3168","PA3574","PA3721","PA3999","PA4003","PA4020","PA4109","PA4110","PA4266","PA4418","PA4522","PA4597","PA4598","PA4599","PA4600","PA4700","PA4776","PA4777","PA4964","PA4967","PA5045","PA5235","PA5471","PA5485","PA5513","PA5514","PA5542","PA0355","PA0357","PA0750","PA1816","PA3002","PA3620","PA4366","PA4400","PA4468","PA4609","PA4946","PA5147","PA5344","PA5443","PA5493"]

output_df = pd.DataFrame(columns = df_columns)

for file in files:
    id_name = (file.columns[0])
    for index, row in file.iterrows():
        new_row = {}
        if row[id_name] not in output_df['ID']:
            new_row["ID"] = row[id_name]
            for field, value in row.items():
                if field[:6] in df_columns:
                    new_row[field[:6]] = str(value)
                    
        output_df = output_df._append(new_row, ignore_index=True)

s = output_df.duplicated(subset=["ID"])
output_df = output_df.assign(duplicated = s)
print(output_df)
print(s)
output_df.to_csv('output.csv')
# print(output_df)
