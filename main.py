from kivymd.app import MDApp
from kivy.uix.screenmanager import Screen,ScreenManager, WipeTransition
from plyer import filechooser,storagepath

class Screen_Manager(ScreenManager):
    pass
class Screen_Login(Screen):
    pass
class Screen_LoginP(Screen):
    pass
class Screen_Home(Screen):
    pass
class Screen_Panel(Screen):
    pass

class mainApp(MDApp):
    def build(self):
        self.theme_cls.theme_style = 'Light'
        self.theme_cls.primary_palette = 'Indigo'
        return 
    def fraud_check(self):
        filechooser.open_file(on_selection  = self.selection_file)
    def selection_file(self,selection):
        self.root.get_screen("Home").ids.state.text = "This will take several mins"
        select = selection[0]
        print(select)
        import pandas as pd
        data = pd.read_csv(select)
        print(data.shape)
        X = data.loc[(data.type == 'TRANSFER') | (data.type == 'CASH_OUT')]

        # shape of x
        X.shape
        X.head()

        X['errorBalanceOrig'] = X.newbalanceOrig + X.amount - X.oldbalanceOrg
        X['errorBalanceDest'] = X.oldbalanceDest + X.amount - X.newbalanceDest

        X = X.drop(['nameDest','nameOrig'], axis = 1)

        # checking the new shape of data
        X.shape
        X['type'].replace('TRANSFER', 0, inplace = True)
        X['type'].replace('CASH_OUT', 1, inplace = True)

        X.isnull().any().any()
        Y = X['isFraud']

        # removing the dependent set
        X = X.drop(['isFraud'], axis = 1)

        # getting the shapes of x and y
        print("Shape of x: ", X.shape)
        print("Shape of y: ", Y.shape)
        from imblearn.over_sampling import SMOTE

        x_resample, y_resample = SMOTE().fit_resample(X, Y.values.ravel())

        # getting the shapes of x and y after resampling
        print("Shape of x: ", x_resample.shape)
        print("Shape of y:", y_resample.shape)

        from sklearn.model_selection import train_test_split

        x_train, x_test, y_train, y_test = train_test_split(x_resample, y_resample, test_size = 0.2, random_state = 0)

        # checking the new shapes
        print("Shape of x_train: ", x_train.shape)
        print("Shape of x_test: ", x_test.shape)
        print("Shape of y_train: ", y_train.shape)
        print("Shape of y_test: ", y_test.shape)

        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler()

        x_train = sc.fit_transform(x_train)
        x_test = sc.transform(x_test)

        from sklearn.metrics import accuracy_score
        import xgboost as xgb
        from xgboost.sklearn import XGBClassifier
        from sklearn.metrics import average_precision_score

        model = XGBClassifier()
        model.load_model("model_sklearn.json")
        # make predictions for test data
        y_pred = model.predict(x_test)
        print(y_pred[0])
        #print(predictions)
        accuracy = accuracy_score(y_test, y_pred)
        accuracy = accuracy * 100.0
        print("Accuracy: %.2f%%" % (accuracy * 100.0))
        text = "you are " + str(accuracy) + " percent secure"
        self.root.get_screen("Home").ids.state.text = text
        

mainApp().run()