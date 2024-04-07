# -*- coding: utf-8 -*-


from google.colab import drive
drive.mount('/content/dhoni')

label_file = r"/content/dhoni/MyDrive/Face_to_BMI/Face-to-height-weight-BMI-estimation--master/Face-to-height-weight-BMI-estimation--master/BMI data - Sheet1.csv"

import pandas as pd

profile_df = pd.read_csv(label_file)

profile_df = pd.read_csv(label_file)

profile_df

"""## How is BMI calculated?
```
BMI = (Weight in Kg) / ((Height in Meters) ^ 2)
```

All the training images are kept in the following directory
"""

data_folder = "/content/dhoni/MyDrive/Face_to_BMI/Face-to-height-weight-BMI-estimation--master/Face-to-height-weight-BMI-estimation--master/height_weight"

from glob import glob
all_files = glob(data_folder+"/*")

all_jpgs = sorted([img for img in all_files if ".jpg" in img or ".jpeg" in img or "JPG" in img])

print("Total {} photos ".format(len(all_jpgs)))

from pathlib import Path as p

def get_index_of_digit(string):
    import re
    match = re.search("\d", p(string).stem)
    return match.start(0)

id_path = [(p(images).stem[:(get_index_of_digit(p(images).stem))],images) for  images in all_jpgs ]

image_df = pd.DataFrame(id_path,columns=['UID','path'])

data_df = image_df.merge(profile_df) ## merged the training images with their profile

data_df

"""## Extract face embedding using facenet pretrained architecture"""

!pip install face_recognition

import face_recognition
import numpy as np

def get_face_encoding(image_path):
    print(image_path)
    picture_of_me = face_recognition.load_image_file(image_path)
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    if not my_face_encoding:
        print("no face found !!!")
        return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()

all_faces = []

for images in data_df.path:
    face_enc = get_face_encoding(images)
    all_faces.append(face_enc)

X = np.array(all_faces) ## This is the training data matrix

y_height = data_df.height.values ## all labels
y_weight = data_df.weight.values
y_BMI = data_df.BMI.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_height_train, y_height_test, y_weight_train, y_weight_test ,y_BMI_train, y_BMI_test = train_test_split(X, y_height,y_weight,y_BMI, random_state=1)

"""## Metric to check the goodness of fit"""

def report_goodness(model,X_test,y_test,predictor_log=True):
    # Make predictions using the testing set
    y_pred = model.predict(X_test)
    y_true = y_test
    if predictor_log:
        y_true = np.log(y_test)
    # The coefficients
    # The mean squared error
    print("Mean squared error: %.2f"      % mean_squared_error(y_true, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_true, y_pred))

    errors = abs(y_pred - y_true)
    mape = 100 * np.mean(errors / y_true)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

"""## Model selection

We will compare few regression model and select the one with better test score . We will compare :

Linear Regression

Ridge Linear Regression

Random Forest Regressor

Kernel Ridge Regressiobn

"""

from sklearn.kernel_ridge import KernelRidge
from sklearn import  linear_model
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

"""# simple linear regression

## Height
"""

model_height = linear_model.LinearRegression()

model_height = model_height.fit(X_train,np.log(y_height_train))

report_goodness(model_height,X_test,y_height_test)

"""We can see the model is only able to explain 39% of the total variance among the variables. There is a room for improvement.

## Weight
"""

model_weight = linear_model.LinearRegression()
model_weight = model_weight.fit(X_train,np.log(y_weight_train))

report_goodness(model_height,X_test,y_weight_test)

"""We can see we have got a negative value of R2. This means the the distnace between the predicted and actual height is very high or in another words our model has underfitted.

## BMI
"""

model_BMI = linear_model.LinearRegression()
model_BMI = model_BMI.fit(X_train,np.log(y_BMI_train))

report_goodness(model_height,X_test,y_BMI_test)

"""We have similar performance in case of BMI as well

## Ridge Linear Regression

## Height
"""

model_height = Ridge(fit_intercept=True, alpha=0.0015, random_state=4)

#model_height = Ridge(fit_intercept=True, alpha=0.0015, random_state=4, normalize=True)

model_height = model_height.fit(X_train,np.log(y_height_train))

report_goodness(model_height,X_test,y_height_test)

"""## Weight"""

model_weight = Ridge(fit_intercept=True, alpha=0.0015, random_state=4)

#model_weight = Ridge(fit_intercept=True, alpha=0.0015, random_state=4, normalize=True)

model_weight = model_weight.fit(X_train,np.log(y_weight_train))

report_goodness(model_weight,X_test,y_weight_test)

"""## BMI"""

model_BMI = Ridge(fit_intercept=True, alpha=0.0015, random_state=4)

#model_BMI = Ridge(fit_intercept=True, alpha=0.0015, random_state=4, normalize=True)

model_BMI = model_BMI.fit(X_train,np.log(y_BMI_train))

report_goodness(model_BMI,X_test,y_BMI_test)

"""## Random Forest Regressor

## Height
"""

model_height = RandomForestRegressor(max_depth=2, random_state=0,
                              n_estimators=100)

model_height = model_height.fit(X_train,np.log(y_height_train))

report_goodness(model_height,X_test,y_height_test)

"""#### With Hyperparameter tuning"""

from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 100, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
rf = RandomForestRegressor()

rf_height_model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_height_model.fit(X_train,np.log(y_height_train))

report_goodness(rf_height_model,X_test,y_height_test)

"""We see our variance score has increased

## Weight
"""

model_weight = RandomForestRegressor(max_depth=2, random_state=0,
                              n_estimators=100)

model_weight = model_weight.fit(X_train,np.log(y_weight_train))

report_goodness(model_weight,X_test,y_weight_test)

"""#### With Hyperparameter tuning"""

rf_weight_model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_weight_model.fit(X_train,np.log(y_weight_train))

report_goodness(rf_weight_model,X_test,y_weight_test)

"""## BMI"""

model_BMI = RandomForestRegressor(max_depth=2, random_state=0,
                              n_estimators=100)

model_BMI = model_BMI.fit(X_train,np.log(y_BMI_train))

report_goodness(model_BMI,X_test,y_BMI_test)

"""#### With Hyperparameter tuning"""

rf_BMI_model = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_BMI_model.fit(X_train,np.log(y_BMI_train))

report_goodness(rf_BMI_model,X_test,y_BMI_test)

"""## Kernel Ridge

## Height
"""

model_height = KernelRidge(kernel='rbf', gamma=0.21,alpha=0.0017)

model_height = model_height.fit(X_train,np.log(y_height_train))

report_goodness(model_height,X_test,y_height_test)

"""## Weight"""

model_weight = KernelRidge(kernel='rbf', gamma=0.21,alpha=0.0017)

model_weight = model_weight.fit(X_train,np.log(y_weight_train))

report_goodness(model_weight,X_test,y_weight_test)

"""## BMI"""

model_BMI = KernelRidge(kernel='rbf', gamma=0.21,alpha=0.0017)

model_BMI = model_BMI.fit(X_train,np.log(y_BMI_train))

report_goodness(model_BMI,X_test,y_BMI_test)

report_goodness(model_BMI,X_test,y_BMI_test)

"""## Conclusion :

We found that kernelised Ridge regression outperformed all the models in terms of the mean squared error and expalined variance

## save all models
"""

!pip install joblib

import joblib

#from sklearn.externals import joblib

"""# save the model to disk"""

height_model = 'weight_predictor.model'
weight_model = 'height_predictor.model'
bmi_model = 'bmi_predictor.model'
joblib.dump(model_height, height_model)
joblib.dump(model_weight, weight_model)
joblib.dump(model_BMI, bmi_model)

"""# load the model from disk"""

height_model = joblib.load(height_model)
weight_model = joblib.load(weight_model)
bmi_model = joblib.load(bmi_model)

"""## test"""

def predict_height_width_BMI(test_image,height_model,weight_model,bmi_model):
    test_array = np.expand_dims(np.array(get_face_encoding(test_image)),axis=0)
    height = np.asscalar(np.exp(height_model.predict(test_array)))
    weight = np.asscalar(np.exp(weight_model.predict(test_array)))
    bmi = np.asscalar(np.exp(bmi_model.predict(test_array)))
    return {'height':height,'weight':weight,'bmi':bmi}

"""## prediction on test images"""

from IPython.display import Image

test_image = r"/content/dhoni/MyDrive/Face_to_BMI/Face-to-height-weight-BMI-estimation--master/Face-to-height-weight-BMI-estimation--master/height_weight_test/akshay1.jpg"
Image(test_image)

def predict_height_width_BMI(test_image, height_model, weight_model, bmi_model):
    # Assuming test_image is the input image data
    height = height_model.predict(test_image)
    weight = weight_model.predict(test_image)
    bmi_input = np.array([[height, weight]])
    bmi = bmi_model.predict(bmi_input)
    return height.item(), weight.item(), bmi.item()

def predict_height_width_BMI(image_path, height_model, weight_model, bmi_model):



    # Predict height and weight
    height = height_model.predict(test_image)
    weight = weight_model.predict(test_image)

    # Create an input array for BMI prediction
    bmi_input = np.array([[height, weight]])

    # Predict BMI
    bmi = bmi_model.predict(bmi_input)

    # Return the predicted height, weight, and BMI
    return height.item(), weight.item(), bmi.item()

predict_height_width_BMI(test_image,height_model,weight_model,bmi_model)

test_image = 'height_weight_test/aamir1.jpg'
Image(test_image)

predict_height_width_BMI(test_image,height_model,weight_model,bmi_model)

test_image = 'height_weight_test/akshay1.jpg'
Image(test_image)

predict_height_width_BMI(test_image,height_model,weight_model,bmi_model)

test_image = 'height_weight_test/ratna1.jpg'
Image(test_image)

predict_height_width_BMI(test_image,height_model,weight_model,bmi_model)

"""## Prediction on some celebrities which is not in training set"""

test_image= 'height_weight_test/jaya1.jpg'
Image(test_image)

predict_height_width_BMI(test_image1,height_model,weight_model,bmi_model)

test_image= 'height_weight_test/hrithik1.jpg'
Image(test_image)

predict_height_width_BMI(test_image1,height_model,weight_model,bmi_model)

test_image= 'height_weight_test/rampal2.jpg'
Image(test_image)

predict_height_width_BMI(test_image1,height_model,weight_model,bmi_model)

test_image= 'height_weight_test/hema1.jpg'
Image(test_image)

predict_height_width_BMI(test_image1,height_model,weight_model,bmi_model)

test_image= 'height_weight_test/jamwal1.jpg'
Image(test_image)

predict_height_width_BMI(test_image1,height_model,weight_model,bmi_model)

test_image= 'height_weight_test/mandira1.jpg'
Image(test_image)

predict_height_width_BMI(test_image1,height_model,weight_model,bmi_model)

"""## Conclusion

Face is almost able to predict BMI correctly. There are places where it is going wrong for example if you predict heigt of really tall celebrity like rampal.
This behavior is acceptable since we have not trained the model on such tall celebrities. Other,issue that i noticed is the face to weight tagging is not that much appropriate since the faces are from diffrent times but the weight is always the latest.
We can get even better model if we have these mappings right.

Also we can improve on height prediction if we model the torso alongwith the face. However , we need to get torso to correct height labeled data for this. In the absence of such data we can rely on the face as height estimator.
"""

