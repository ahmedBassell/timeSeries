# Linear regression

from sklearn import linear_model
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

import numpy as np
import pandas as pd
import os

# read db
db_path = os.path.join(os.path.dirname(__file__), 'db/newdb.csv')
df = pd.read_csv(db_path, sep=',', dtype={'id':np.int64, 'hour':np.int64, 'camid': np.int64})



# enc = preprocessing.OneHotEncoder()

# enc.fit([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],[0, 0, 1, 0, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0],[0, 0, 0, 0, 1, 0, 0],[0, 0, 0, 0, 0, 1, 0],[0, 0, 0, 0, 0, 0, 1],])  

# OneHotEncoder(categorical_features='day', dtype=<'str'>,handle_unknown='error', n_values='auto', sparse=True)

# enc.transform([[0, 1, 3]]).toarray()




from sklearn.feature_extraction import DictVectorizer

def one_hot_dataframe(data, cols, replace=False):
    vec = DictVectorizer()
    mkdict = lambda row: dict((col, row[col]) for col in cols)
    vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
    vecData.columns = vec.get_feature_names()
    vecData.index = data.index
    if replace is True:
        data = data.drop(cols, axis=1)
        data = data.join(vecData)
    return (data, vecData, vec)

# data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
#         'year': [2000, 2001, 2002, 2001, 2002],
#         'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}

# df = pd.DataFrame(data)

df2, _, _ = one_hot_dataframe(df, ['day'], replace=True)
# print df2
df = df2






def convert_to_enc(row):
	week_enc = {'Sunday':2, 'Monday':3, 'Tuesday':4, 'Wednesday': 5, 'Thursday': 6, 'Friday': 7, 'Saturday': 1}
	row['enc_day'] = week_enc[row['day']]
	return row
def conv_month(row):
	row['Month'] = pd.to_datetime(row['date']).month
	return row

# df = df.apply(convert_to_enc, axis=1)

# df =  df.apply(conv_month, axis=1)

# feature and target
x = df[['Date', 'day=Friday', 'day=Saturday', 'day=Sunday', 'day=Monday', 'day=Tuesday', 'day=Wednesday', 'day=Thursday']]
y = df['count'].as_matrix().reshape(-1,1)


# Split the data into training/testing sets
x_train = x[:-14]
x_test = x[-14:]

# Split the targets into training/testing sets
y_train = y[:-14]
y_test = y[-14:]


# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)




# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean square error
print("Residual sum of squares: %.2f"
      % np.mean((regr.predict(x_test) - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x_test, y_test))

# Plot outputs
plt.scatter(x_test['Date'], y_test,  color='black')
plt.plot(x_test['Date'], regr.predict(x_test), color='blue',
         linewidth=3)



plt.show()


plt.scatter(x['Date'], y,  color='black')
plt.plot(x['Date'], regr.predict(x), color='blue',linewidth=3)


plt.show()




def pred_nex_week(regr, start_date = 736136, wday = 4):
	wdays = ['Friday', 'Saturday', 'Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday']
	pred_days_in = []
	pred_week_in = []
	for i in range(7):
		dpos = (wday +i) % 7
		d_count = start_date + i
		pred_days_in.append(d_count)
		pred_day_week = [float(j==dpos) for j in range(7)]
		pred_day = [d_count]
		pred_day.extend(pred_day_week)
		pred_week_in.append(pred_day)
	pred_week = regr.predict(pred_week_in)
	pred_week = pred_week.tolist()
	return pred_days_in,pred_week

def to_1d(arr):
	newarr = []
	for i in arr:
		x = i[0]
		newarr.append(x)
	return newarr


ext_x, ext_y =  pred_nex_week(regr)

# fig, ax = plt.subplots()
# rects1 = ax.bar(0.34, ext_y, .23, color='r')
# 
# 
all_x = x['Date'].as_matrix().tolist()
all_y = y.tolist()


all_x.extend(ext_x)
all_y.extend(ext_y)
all_y = to_1d(all_y)
# print len(all_x)
# print type(all_x[61])
# print type(all_y[61])
# print ext_x
fig, ax = plt.subplots()

trafficBar = ax.bar(all_x, all_y, width=1, label="hi", color=['green'], alpha=0.5)
for i in range(7):
	trafficBar[-(i+1)].set_color('red')
# assign locator and formatter for the xaxis ticks.
ax.xaxis.set_major_locator(mdates.AutoDateLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y.%m.%d'))
ax.xaxis_date()
fig.autofmt_xdate()

plt.xlabel('Group')
plt.ylabel('Scores')
plt.title('Scores by group and gender')
plt.legend()

plt.tight_layout()
plt.show()