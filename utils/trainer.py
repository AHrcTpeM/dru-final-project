from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return LinearDiscriminantAnalysis().fit(train_x, train_y)

    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)