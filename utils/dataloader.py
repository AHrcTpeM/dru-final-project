import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder


class DataLoader(object):
    def fit(self, dataset):
        self.dataset = dataset.copy()

    def load_data(self):
        # columns combination
        self.dataset['Insurance_Driving_Risk'] = self.dataset['Driving_License'] * self.dataset['Previously_Insured']
        self.dataset['Premium_Per_Day'] = self.dataset['Annual_Premium'] / self.dataset['Vintage']

        # log transform value
        self.dataset['Premium_Per_Day'] = np.log(self.dataset['Premium_Per_Day'] + 1)

        # binning with qcut
        self.dataset['Annual_Premium'] = pd.qcut(self.dataset['Annual_Premium'], 4)

        # binning with cut
        self.dataset['Age'] = pd.cut(self.dataset['Age'], 5)

        # drop columns
        drop_elements = ['id']
        self.dataset = self.dataset.drop(drop_elements, axis=1)

        # encode labels
        le = LabelEncoder()

        le.fit(self.dataset['Gender'])
        self.dataset['Gender'] = le.transform(self.dataset['Gender'])

        le.fit(self.dataset['Vehicle_Age'])
        self.dataset['Vehicle_Age'] = le.transform(self.dataset['Vehicle_Age'])

        le.fit(self.dataset['Vehicle_Damage'])
        self.dataset['Vehicle_Damage'] = le.transform(self.dataset['Vehicle_Damage'])

        le.fit(self.dataset['Age'])
        self.dataset['Age'] = le.transform(self.dataset['Age'])

        le.fit(self.dataset['Annual_Premium'])
        self.dataset['Annual_Premium'] = le.transform(self.dataset['Annual_Premium'])

        return self.dataset