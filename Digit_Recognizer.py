import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv("C:\\Users\\DELL\\Desktop\\Excelr Hyd\\Kaggle\\Digit Recognizer\\train.csv").as_matrix()

test_data=pd.read_csv("C:\\Users\\DELL\\Desktop\\Excelr Hyd\\Kaggle\\Digit Recognizer\\test.csv").as_matrix()

x_train=train[:,1:]
y_train=train[:,0]


from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=50,criterion="entropy")
model.fit(x_train,y_train)

pred=model.predict(test_data)


preds=pd.DataFrame(pred)
sub_df=pd.read_csv("C:\\Users\\DELL\\Desktop\\Excelr Hyd\\Kaggle\\Digit Recognizer\\sample_submission.csv")
datasets=pd.concat([sub_df["ImageId"],preds],axis=1)
datasets.columns=["ImageId","Label"]
datasets.to_csv("Digit_submission.csv",index=False

# Kaggle Score for this model is 0.96371
