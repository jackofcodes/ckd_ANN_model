
# coding: utf-8

# In[79]:


import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[84]:


df1 = pd.read_csv('D:\B TECH\SEM 6\SOFTWARE ENGINEERING\ANN_Model\ckd2.csv')


# In[85]:


df1.head()


# In[88]:


from sklearn.impute import SimpleImputer

# Replace "?" with NaN
df1.replace("?", np.nan, inplace=True)

# Define columns needing imputation
columns_for_imputation = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
                          'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 
                          'wbcc', 'rbcc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

# Initialize the imputer for numeric columns
imputer_numeric = SimpleImputer(strategy='mean')

# Impute missing values for numeric columns
numeric_cols = df1.select_dtypes(include=['float64', 'int64']).columns
df1[numeric_cols] = imputer_numeric.fit_transform(df1[numeric_cols])

# Initialize the imputer for categorical columns
imputer_categorical = SimpleImputer(strategy='most_frequent')

# Impute missing values for categorical columns
categorical_cols = df1.select_dtypes(include=['object']).columns
df1[categorical_cols] = imputer_categorical.fit_transform(df1[categorical_cols])

# Now check for missing values again
missing_values = df1.isnull().sum()
print(missing_values)


# In[90]:


sns.heatmap(df1.isnull(), yticklabels = False, cbar = False, cmap="Blues")


# In[91]:


df1.info()


# In[92]:


print(df1.columns)


# In[93]:


df1.shape


# In[94]:


total_null_values = df1.isnull().sum().sum()
print("\nTotal null values in the DataFrame:", total_null_values)


# In[95]:


df1.head()


# In[96]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for col in categorical_cols:
    df1[col] = label_encoder.fit_transform(df1[col])


# In[97]:


df1.tail()


# In[98]:


sns.scatterplot(df1)


# In[99]:


corr_matrix = df1.corr()

# Create a heatmap using seaborn with figsize parameter
plt.figure(figsize=(30,30))
sns.heatmap(corr_matrix, annot=True)

# Save the plot as an image
plt.savefig("correlation_heatmap.png")

# Close the plot to avoid blocking the script execution
plt.close()

print("Correlation heatmap saved as correlation_heatmap.png")


# In[100]:


df1.columns = df1.columns.str.replace("'", "")

# Check if the column 'class' exists after removing quotation marks
if 'class' in df1.columns:
    y = df1['class']
else:
    print("'class' column not found after removing quotation marks.")


# In[101]:


# Selecting specific columns from the original DataFrame
selected_columns = ['age', 'sg', 'al', 'sc', 'sod', 'pcv', 'htn', 'dm']

# Create a new DataFrame containing only the selected columns
x = df1[selected_columns].copy()

# Print the first few rows of the new DataFrame to verify
x.head()


# In[102]:


y = df1['class'].copy()

# Print the first few rows of the new DataFrame to verify
y.head()


# In[103]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)


# In[104]:


from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


# In[105]:


import numpy as np
x_train = np.array(x_train).astype('float32')
y_train = np.array(y_train).astype('float32')
x_test = np.array(x_test).astype('float32')
y_test = np.array(y_test).astype('float32')
import warnings
warnings.filterwarnings('ignore')


# In[106]:


x_train.shape


# In[107]:


y_train.shape


# In[108]:


ANN_model = keras.Sequential()
ANN_model.add(Dense(50, input_dim=8))
ANN_model.add(Activation('relu'))
ANN_model.add(Dense(150))
ANN_model.add(Activation('relu'))
ANN_model.add(Dense(1, activation='sigmoid'))  # Changed to sigmoid activation
ANN_model.compile(loss='binary_crossentropy', optimizer='adam')  # Changed loss function
ANN_model.summary()


# In[109]:


ANN_model.compile(optimizer='Adam', loss='binary_crossentropy')
epochs_hist = ANN_model.fit(x_train, y_train, epochs = 100 , batch_size = 30 , validation_split = 0.2)


# In[110]:


result = ANN_model.evaluate(x_test , y_test)
accuracy_ANN = 1-result
print('Accuracy: {}'.format(accuracy_ANN))


# In[111]:


y_pred_probs = ANN_model.predict(x_test)

# Convert predicted probabilities to class labels using argmax
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# If y_test is a Pandas Series, you may need to convert it to a NumPy array for comparison
y_test_array = y_test.values if isinstance(y_test, pd.Series) else y_test


# In[112]:


from sklearn.metrics import confusion_matrix,classification_report

cm=confusion_matrix(y_test,y_test_array)
print(cm)

ANN_model.save(r"D:\B TECH\SEM 6\SOFTWARE ENGINEERING\ANN_Model\model.h5")
print("Model saved successfully.")








