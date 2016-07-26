
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import cross_validation



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



# feature and target
x_d = df['i']
# x_d = df['i']
y_d = df['count']


# Split the data into training/testing sets
x_train = x_d[:-14]
x_test = x_d[-14:]

# Split the targets into training/testing sets
y_train = y_d[:-14]
y_test = y_d[-14:]















np.random.seed(0)

n_samples = 30
degrees = [20]

# true_fun = lambda X: np.cos(1.5 * np.pi * X)
# X = np.sort(np.random.rand(n_samples))
# y = true_fun(X) + np.random.randn(n_samples) * 0.1
X = x_d.as_matrix()
y = y_d.as_matrix()
print X
print y

plt.figure(figsize=(18, 8))
for i in range(len(degrees)):
    ax = plt.subplot(1, len(degrees), i + 1)
    plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i],
                                             include_bias=True)
    linear_regression = LinearRegression()
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    pipeline.fit(X[:, np.newaxis], y)

    # Evaluate the models using crossvalidation
    scores = cross_validation.cross_val_score(pipeline,
        X[:, np.newaxis], y, scoring="mean_squared_error", cv=10)

    X_test = x_test
    plt.plot(X, pipeline.predict(X[:, np.newaxis]), label="Model")
    # plt.plot(X_test, true_fun(X_test), label="True function")
    plt.scatter(X, y, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((-1, 60))
    plt.ylim((0, 20000))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(
        degrees[i], -scores.mean(), scores.std()))
plt.show()