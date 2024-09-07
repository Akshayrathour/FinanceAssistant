from  keras.models import load_model
import pandas as pd
import numpy as np
import datetime
import yfinance as yf

model = load_model(r'C:\Users\Aksha\Downloads\stmodel.h5')
# df=pd.read_csv(r'C:\Users\Aksha\Downloads\NFLX.csv')
start=datetime.datetime(2023,12,1)
end=datetime.datetime(2024,1,1)
df=yf.download('MSFT',start,end)


# def str_to_datetime(s):
#   split = s.split('-')
#   year, month, day = int(split[0]), int(split[1]), int(split[2])
#   return datetime.datetime(year=year, month=month, day=day)

# def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
#   first_date = str_to_datetime(first_date_str)
#   last_date  = str_to_datetime(last_date_str)

#   target_date = first_date

#   dates = []
#   X, Y = [], []
#   last_time = False
#   while True:
#     df_subset = dataframe.loc[:target_date].tail(n+1)
#     # print(df_subset)
#     if len(df_subset) != n+1:
#       print(f'Error: Window of size {n} is too large for date {target_date}')
#       return

#     values = df_subset['Close'].to_numpy()
#     x, y = values[:-1], values[-1]

#     dates.append(target_date)
#     X.append(x)
#     Y.append(y)

#     next_week = dataframe.loc[target_date:target_date+datetime.timedelta(days=7)]
#     next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
#     next_date_str = next_datetime_str.split('T')[0]
#     year_month_day = next_date_str.split('-')
#     year, month, day = year_month_day
#     next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

#     if last_time:
#       break

#     target_date = next_date

#     if target_date == last_date:
#       last_time = True

#   ret_df = pd.DataFrame({})
#   ret_df['Target Date'] = dates

#   X = np.array(X)
#   for i in range(0, n):
#     X[:, i]
#     ret_df[f'Target-{n-i}'] = X[:, i]

#   ret_df['Target'] = Y

#   return ret_df

# # Start day second time around: '2021-03-25'
# datetime_object = str_to_datetime('2023-12-01')

# windowed_df = df_to_windowed_df(df,'2023-12-01','2024-01-01',n=3)

# def windowed_df_to_date_X_y(windowed_dataframe):
#   df_as_np = windowed_dataframe.to_numpy()

#   dates = df_as_np[:, 0]

#   middle_matrix = df_as_np[:, 1:-1]
#   X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

#   Y = df_as_np[:, -1]

#   return dates, X.astype(np.float32), Y.astype(np.float32)

# dates, X, y = windowed_df_to_date_X_y(windowed_df)

df=df.iloc[:,-3:]
df=df.to_numpy()
df=df[-3:]
print(df)
res=model.predict(df)
print(res)
