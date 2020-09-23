import pandas as pd
import os.path

data = [1,2,3]

def get_df_from_db(sql):
    return pd.DataFrame(data,index=range(len(data)))

crm_sql = ['sql1','sql2','sql3']
crm_list = ['crm_employee','crm_product','crm_promotion']
for i in range(len(crm_sql)):
    path = r'D:\test\data'
    dir = os.path.join(path, '{}.csv').format(crm_list[i])
    data = get_df_from_db(crm_sql[i])
    print(dir)
    data.to_csv(dir, index=0, header=1)