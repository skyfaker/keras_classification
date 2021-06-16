import pandas as pd
from configparser import ConfigParser
import numpy as np


print("读取配置。。。")
cfg = ConfigParser()
cfg.read('config.ini', encoding='UTF-8')

print("读取excel文件。。。")
excel_name = cfg.get('excel文件名','name')

xls = pd.ExcelFile(excel_name)

day_time = cfg.get('日报','time')

service_hall = ["中国电信上海昌里东路营业厅",
                "易贝上海北艾路天翼专营店",
                "共联上海永泰路营业厅",
                "易贝上海和炯路天翼专营店",
                "易贝上海杨思路营业厅",
                "鹏德上海永泰路营业厅",
                "宁皙上海峨山路天翼专营店",
                "汉启上海东三里桥路天翼专营店",
                "易贝上海凌兆路天翼专营店",
                "汉启上海长清路天翼专营店"
               ]

worker_num = ["DLS_HQTX_SPR", "dls_hwdx_ywr"]

raw_df = pd.DataFrame(index = service_hall)

print("正在生成日报。。。")
# 宽带
sheet_name = '宽带'
df = pd.read_excel(xls, sheet_name)

df = df[df["客户类型"] == "公众客户"]
df = df[df["发展部门"].isin(service_hall)]

df['完工日期'] = pd.to_datetime(df['完工日期'])
df = df[df["完工日期"] == day_time]

ds = df["发展部门"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展部门":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# WiFi
sheet_name = 'WiFi'
df = pd.read_excel(xls, sheet_name)
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展门店"].isin(service_hall)]

df['促销时间'] = pd.to_datetime(df['促销时间'])
df = df[df["促销时间"] == day_time]

ds = df["发展门店"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展门店":"WiFi"}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# 重点号卡
sheet_name = '重点号卡'
df = pd.read_excel(xls, sheet_name)
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展部门"].isin(service_hall)]

df['统计日期'] = pd.to_datetime(df['统计日期'])
df = df[df["统计日期"] == day_time]

df = df[df["服务提供小类"] == "携入新装"]

ds = df["发展部门"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展部门":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# 5G1
sheet_name = '5G1'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[df["首次5G时间"] == day_time]
df = df[df["主副卡"] == "主卡"]

df = df[df["发展门店"].isin(service_hall)]
ds = df["发展门店"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展门店":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# 5G2
sheet_name = '5G2'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[df["首次5G时间"] == day_time]
df = df[df["主副卡"] == "主卡"]
tmp = df.copy()

df = df[df["发展部门"].isin(service_hall)]
ds = df["发展部门"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展部门":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)
raw_df.sort_index(inplace=True, ascending = False)

# 5G
raw_df.fillna(0, inplace=True)
raw_df['5G'] = raw_df['5G1'] + raw_df['5G2']
raw_df.loc["sum"] = raw_df.apply(lambda x:x.sum())


# 工号
worker_df = pd.DataFrame(index = worker_num)
empty_col = ["宽带", "WiFi", "重点号卡"]
worker_df[empty_col] = pd.DataFrame([[np.nan, np.nan, np.nan]], index=worker_df.index)

# 工号：5G1
sheet_name = '5G1'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[df["首次5G时间"] == day_time]
df = df[df["主副卡"] == "主卡"]

df = df[df["工号"].isin(worker_num)]
ds = df["工号"].value_counts()

re_df = pd.DataFrame(ds)
re_df.rename(columns={"工号":sheet_name}, inplace = True)

worker_df = pd.concat([worker_df, re_df], axis=1, sort =True)

# 工号：5G2
sheet_name = '5G2'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[df["首次5G时间"] == day_time]
df = df[df["主副卡"] == "主卡"]

df = df[df["工号"].isin(worker_num)]
ds = df["工号"].value_counts()

re_df = pd.DataFrame(ds)
re_df.rename(columns={"工号":sheet_name}, inplace = True)

worker_df = pd.concat([worker_df, re_df], axis=1, sort =True)

worker_df.fillna(0, inplace=True)
worker_df['5G'] = worker_df['5G1'] + worker_df['5G2']
worker_df.loc["worker_sum"] = worker_df.apply(lambda x:x.sum())

# 合并
raw_df = raw_df.append(worker_df, sort=False)

# 后处理
for col in raw_df.columns:
    raw_df[col] = raw_df[col].astype(np.int16)

# print(raw_df.shape)
print("日报生成完成！")
print()

raw_df.to_csv("日报.csv", encoding='utf_8_sig')
del raw_df


####################################################
# 周报
print("正在生成周报。。。")

week_start = cfg.get('周报','start')
week_end = cfg.get('周报','end')

raw_df = pd.DataFrame(index = service_hall)

# 宽带
sheet_name = '宽带'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展部门"].isin(service_hall)]

df['完工日期'] = pd.to_datetime(df['完工日期'])
df = df[(df['完工日期'] >=pd.to_datetime(week_start)) & (df['完工日期'] <= pd.to_datetime(week_end))]

ds = df["发展部门"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展部门":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# WiFi
sheet_name = 'WiFi'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展门店"].isin(service_hall)]

df['促销时间'] = pd.to_datetime(df['促销时间'])
df = df[(df['促销时间'] >=pd.to_datetime(week_start)) & (df['促销时间'] <= pd.to_datetime(week_end))]

ds = df["发展门店"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展门店":"WiFi"}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# 重点号卡
sheet_name = '重点号卡'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展部门"].isin(service_hall)]

df['统计日期'] = pd.to_datetime(df['统计日期'])
df = df[(df['统计日期'] >=pd.to_datetime(week_start)) & (df['统计日期'] <= pd.to_datetime(week_end))]

df = df[df["服务提供小类"] == "携入新装"]

ds = df["发展部门"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展部门":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# 5G1
sheet_name = '5G1'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展门店"].isin(service_hall)]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[(df['首次5G时间'] >=pd.to_datetime(week_start)) & (df['首次5G时间'] <= pd.to_datetime(week_end))]

df = df[df["主副卡"] == "主卡"]

ds = df["发展门店"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展门店":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# 5G2
sheet_name = '5G2'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展部门"].isin(service_hall)]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[(df['首次5G时间'] >=pd.to_datetime(week_start)) & (df['首次5G时间'] <= pd.to_datetime(week_end))]

df = df[df["主副卡"] == "主卡"]

ds = df["发展部门"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展部门":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)
raw_df.sort_index(inplace=True, ascending = False)

# 5G
raw_df.fillna(0, inplace=True)
raw_df['5G'] = raw_df['5G1'] + raw_df['5G2']
raw_df.loc["sum"] = raw_df.apply(lambda x:x.sum())

# 工号
worker_df = pd.DataFrame(index = worker_num)
empty_col = ["宽带", "WiFi", "重点号卡"]
worker_df[empty_col] = pd.DataFrame([[np.nan, np.nan, np.nan]], index=worker_df.index)

# 工号：5G1
sheet_name = '5G1'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[(df['首次5G时间'] >=pd.to_datetime(week_start)) & (df['首次5G时间'] <= pd.to_datetime(week_end))]

df = df[df["主副卡"] == "主卡"]

df = df[df["工号"].isin(worker_num)]
ds = df["工号"].value_counts()

re_df = pd.DataFrame(ds)
re_df.rename(columns={"工号":sheet_name}, inplace = True)

worker_df = pd.concat([worker_df, re_df], axis=1, sort =True)

# 工号：5G2
sheet_name = '5G2'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[(df['首次5G时间'] >=pd.to_datetime(week_start)) & (df['首次5G时间'] <= pd.to_datetime(week_end))]

df = df[df["主副卡"] == "主卡"]

df = df[df["工号"].isin(worker_num)]
ds = df["工号"].value_counts()

re_df = pd.DataFrame(ds)
re_df.rename(columns={"工号":sheet_name}, inplace = True)

worker_df = pd.concat([worker_df, re_df], axis=1, sort =True)

worker_df.fillna(0, inplace=True)
worker_df['5G'] = worker_df['5G1'] + worker_df['5G2']
worker_df.loc["worker_sum"] = worker_df.apply(lambda x:x.sum())

# 合并
raw_df = raw_df.append(worker_df, sort=False)

# 后期处理
for col in raw_df.columns:
    raw_df[col] = raw_df[col].astype(np.int16)

# print(raw_df.shape)
print("周报生成完成！")
print()

raw_df.to_csv("周报.csv", encoding='utf_8_sig')
del raw_df



####################################################
# 月报
print("正在生成月报。。。")

month_start = cfg.get('月报','start')
month_end = cfg.get('月报','end')

raw_df = pd.DataFrame(index = service_hall)

# 宽带
sheet_name = '宽带'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展部门"].isin(service_hall)]

df['完工日期'] = pd.to_datetime(df['完工日期'])
df = df[(df['完工日期'] >=pd.to_datetime(month_start)) & (df['完工日期'] <= pd.to_datetime(month_end))]

ds = df["发展部门"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展部门":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# WiFi
sheet_name = 'WiFi'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展门店"].isin(service_hall)]

df['促销时间'] = pd.to_datetime(df['促销时间'])
df = df[(df['促销时间'] >=pd.to_datetime(month_start)) & (df['促销时间'] <= pd.to_datetime(month_end))]

ds = df["发展门店"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展门店":"WiFi"}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# 重点号卡
sheet_name = '重点号卡'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展部门"].isin(service_hall)]

df['统计日期'] = pd.to_datetime(df['统计日期'])
df = df[(df['统计日期'] >=pd.to_datetime(month_start)) & (df['统计日期'] <= pd.to_datetime(month_end))]

df = df[df["服务提供小类"] == "携入新装"]

ds = df["发展部门"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展部门":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# 5G1
sheet_name = '5G1'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展门店"].isin(service_hall)]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[(df['首次5G时间'] >=pd.to_datetime(month_start)) & (df['首次5G时间'] <= pd.to_datetime(month_end))]

df = df[df["主副卡"] == "主卡"]

ds = df["发展门店"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展门店":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)


# 5G2
sheet_name = '5G2'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]
df = df[df["发展部门"].isin(service_hall)]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[(df['首次5G时间'] >=pd.to_datetime(month_start)) & (df['首次5G时间'] <= pd.to_datetime(month_end))]

df = df[df["主副卡"] == "主卡"]

ds = df["发展部门"].value_counts()
re_df = pd.DataFrame(ds)
re_df.rename(columns={"发展部门":sheet_name}, inplace = True)

raw_df = pd.concat([raw_df, re_df], axis=1, sort =True)
raw_df.sort_index(inplace=True, ascending = False)

# 5G
raw_df.fillna(0, inplace=True)
raw_df['5G'] = raw_df['5G1'] + raw_df['5G2']
raw_df.loc["sum"] = raw_df.apply(lambda x:x.sum())


# 工号
worker_df = pd.DataFrame(index = worker_num)
empty_col = ["宽带", "WiFi", "重点号卡"]
worker_df[empty_col] = pd.DataFrame([[np.nan, np.nan, np.nan]], index=worker_df.index)

# 工号：5G1
sheet_name = '5G1'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[(df['首次5G时间'] >=pd.to_datetime(month_start)) & (df['首次5G时间'] <= pd.to_datetime(month_end))]

df = df[df["主副卡"] == "主卡"]

df = df[df["工号"].isin(worker_num)]
ds = df["工号"].value_counts()

re_df = pd.DataFrame(ds)
re_df.rename(columns={"工号":sheet_name}, inplace = True)

worker_df = pd.concat([worker_df, re_df], axis=1, sort =True)

# 工号：5G2
sheet_name = '5G2'
df = pd.read_excel(xls, sheet_name )
df = df[df["客户类型"] == "公众客户"]

df['首次5G时间'] = pd.to_datetime(df['首次5G时间'])
df = df[(df['首次5G时间'] >=pd.to_datetime(month_start)) & (df['首次5G时间'] <= pd.to_datetime(month_end))]

df = df[df["主副卡"] == "主卡"]

df = df[df["工号"].isin(worker_num)]
ds = df["工号"].value_counts()

re_df = pd.DataFrame(ds)
re_df.rename(columns={"工号":sheet_name}, inplace = True)

worker_df = pd.concat([worker_df, re_df], axis=1, sort =True)
worker_df.fillna(0, inplace=True)

worker_df['5G'] = worker_df['5G1'] + worker_df['5G2']

worker_df.loc["worker_sum"] = worker_df.apply(lambda x:x.sum())

# 合并
raw_df = raw_df.append(worker_df, sort=False)

# 后处理
for col in raw_df.columns:
    raw_df[col] = raw_df[col].astype(np.int16)


# print(raw_df.shape)
print("月报生成完成！")
print()

raw_df.to_csv("月报.csv", encoding='utf_8_sig')