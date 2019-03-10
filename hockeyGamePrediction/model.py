import numpy
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.preprocessing.text import Tokenizer
# create the tokenizer
t = Tokenizer()

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load dataset
my_data = pd.read_csv("players_as_numbers.csv")

my_data = my_data.dropna()

my_data = my_data.drop('Unnamed: 0', axis=1)

dataset = my_data.values

X = dataset[:,0:38]
Y = dataset[:,38]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# baseline model
def create_baseline():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='softmax'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline, epochs=1000, batch_size=5, verbose=1)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
