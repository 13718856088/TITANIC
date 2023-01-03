import functools

import tensorflow as tf
import numpy as np
import pandas as pd

#df = pd.read_csv('train.csv')
#x_data = np.array(df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch',
                     # 'Ticket', 'Fare', 'Cabin', 'Embarked']])
#y_data = np.array(df[['Survived']])

'''df = pd.read_csv('test.csv')
df['Survived'] = None
df.to_csv('test.csv')'''

labelColumn = "Survived"  # 指定数据标签的列名
labels = [0, 1]


def getDataset(filePath, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(filePath,
                                                    batch_size=10,
                                                    label_name=labelColumn,
                                                    na_value="?",
                                                    num_epochs=1,
                                                    ignore_errors=True,
                                                    **kwargs)
    return dataset


rawTrainData = getDataset("train.csv")
rawTestData = getDataset("test.csv")

#print(rawTrainData)


def showBatch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}:{}".format(key, value.numpy()))
        print("{:20s}:{}".format("label", label.numpy()))


#showBatch(rawTrainData)

# 处理连续性数据
'''selectColumns = ['Survived', 'Age', 'SibSp', 'Parch', 'Fare']
defaults = [0, 0.0, 0.0, 0.0, 0.0]
tempDataset = getDataset('train.csv',
                         select_columns = selectColumns,
                         column_defaults = defaults)'''

# showBatch(tempDataset)
# example_batch, labels_batch = next(iter(tempDataset))


# 将所有列打包到一起
def pack(features, label):
    return tf.stack(list(features.values()),axis=1),label


# packed_dataset = tempDataset.map(pack)

# for fetures, labels in packed_dataset.take(1):
   # print(fetures.numpy(),labels.numpy(),sep="\n\n")


class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=1)
        features["numeric"] = numeric_features

        return features, labels


numericFeatures = ['Age', 'SibSp', 'Parch', 'Fare']

packed_train_data = rawTrainData.map(PackNumericFeatures(numericFeatures))
packed_test_data = rawTestData.map(PackNumericFeatures(numericFeatures))

# showBatch(packed_train_data)
# showBatch(packed_test_data)

example_batch, labels_batch = next(iter(packed_train_data))

desc = pd.read_csv('train.csv')[numericFeatures].describe()

# print(desc)

mean = np.array(desc.T["mean"])
std = np.array(desc.T["std"])

'''print(mean, type(mean))
print(std, type(std))'''


def normalization(data, mean1, std1):
    return (data - mean)/std


normalizer = functools.partial(normalization, mean1=mean, std1=std)
numeric_column = tf.feature_column.numeric_column("numeric", normalizer_fn=normalizer, shape=[len(numericFeatures)])
numeric_columns = [numeric_column]

# print(numeric_column)

# print(example_batch["numeric"])

'''numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
print(numeric_layer(example_batch).numpy())'''

CATEGORIES = {

    'Pclass': [1, 2, 3],
    'Sex': ['male', 'female'],
    'Embarked' : ['S', 'C', 'Q']
}

categorical_columns = []
for feature, vocab in CATEGORIES.items():
  cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
  categorical_columns.append(tf.feature_column.indicator_column(cat_col))

# print(categorical_columns)

'''categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print(categorical_layer(example_batch).numpy()[0])'''

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)

# print(preprocessing_layer(example_batch).numpy()[0])


model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1, activation='sigmoid')
   # tf.keras.layers.Dense(1)
])

model.compile(
# loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    loss='binary_crossentropy',
    optimizer="adam",
    metrics=["accuracy"])

train_data = packed_train_data.shuffle(500)
test_data = packed_test_data

model.fit(train_data, epochs=40)

p_survive = model.predict(test_data)
p_survive = (p_survive > 0.5).astype(int)
print(p_survive)

sub = pd.read_csv('gender_submission.csv')
sub['Survived'] = list(map(int, p_survive))
sub.to_csv('submission.csv', index=False)