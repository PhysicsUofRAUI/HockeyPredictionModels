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
from keras.optimizers import SGD
opt = SGD(lr=100)

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

print("Number,results,epochs,batch_size,number of layers,depth")




# first model
def create_baseline_1():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_1, epochs=500, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,500,25,10,50" % (results.mean()*100, results.std()*100))



# second model
def create_baseline_2():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_2, epochs=500, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,500,25,15,50" % (results.mean()*100, results.std()*100))





# third model
def create_baseline_3():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_3, epochs=500, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,500,25,20,50" % (results.mean()*100, results.std()*100))





# fourth model
def create_baseline_4:
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_4, epochs=500, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,500,25,25,50" % (results.mean()*100, results.std()*100))





# next set has 100 neurons per layer




# fifth model
def create_baseline_5():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_5, epochs=500, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,500,25,10,100" % (results.mean()*100, results.std()*100))



# sixth model
def create_baseline_6():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_6, epochs=500, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,500,25,15,50" % (results.mean()*100, results.std()*100))





# seventh model
def create_baseline_7():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_7, epochs=500, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,500,25,20,100" % (results.mean()*100, results.std()*100))





# eight model
def create_baseline_8():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_3, epochs=500, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,500,25,25,100" % (results.mean()*100, results.std()*100))









# these next ones are repeat but have 2000 epochs









# first model
def create_baseline_1():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_1, epochs=2000, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,2000,25,10,50" % (results.mean()*100, results.std()*100))



# second model
def create_baseline_2():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_2, epochs=2000, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,2000,25,15,50" % (results.mean()*100, results.std()*100))





# third model
def create_baseline_3():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_3, epochs=2000, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,2000,25,20,50" % (results.mean()*100, results.std()*100))





# fourth model
def create_baseline_4:
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(50, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_4, epochs=2000, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,2000,25,25,50" % (results.mean()*100, results.std()*100))





# next set has 100 neurons per layer




# fifth model
def create_baseline_5():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_5, epochs=2000, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,2000,25,10,100" % (results.mean()*100, results.std()*100))



# sixth model
def create_baseline_6():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_6, epochs=2000, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,2000,25,15,50" % (results.mean()*100, results.std()*100))





# seventh model
def create_baseline_7():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_7, epochs=2000, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,2000,25,20,100" % (results.mean()*100, results.std()*100))





# eight model
def create_baseline_8():
	# create model
	model = Sequential()
	model.add(Dense(38, input_dim=38, kernel_initializer='normal', activation='relu'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(100, kernel_initializer='normal', activation='sigmoid'))
	model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adagrad', metrics=['accuracy'])
	return model

# evaluate model with standardized dataset
estimator = KerasClassifier(build_fn=create_baseline_3, epochs=2000, batch_size=25, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(estimator, X, encoded_Y, cv=kfold)
print("1, %.2f%% , (%.2f%%) ,2000,25,25,100" % (results.mean()*100, results.std()*100))
