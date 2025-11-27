import numpy as np
import matplotlib.pyplot as plt
import time
import math
import arff

class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []

        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]

        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator


def train_test_split(X, y, test_size=0.2, random_state=42):
    if random_state is not None:
        np.random.seed(random_state)
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_test = math.ceil(n_samples * test_size)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    if X.ndim == 1:
        X_train, X_test = X[train_indices], X[test_indices]
    else:
        X_train, X_test = X[train_indices, :], X[test_indices, :]

    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test


def carregar_arff():
    dataset = arff.load(open("obesity.arff", "r"))
    dados = np.array(dataset["data"], dtype=object)

    Gender = dados[:, 0]
    Age = dados[:, 1]
    Height = dados[:, 2]
    Weight = dados[:, 3]
    family_history_with_overweight = dados[:, 4]
    FAVC = dados[:, 5]
    FCVC = dados[:, 6]
    NCP = dados[:, 7]
    CAEC = dados[:, 8]
    SMOKE = dados[:, 9]
    CH2O = dados[:, 10]
    SCC = dados[:, 11]
    FAF = dados[:, 12]
    TUE = dados[:, 13]
    CALC = dados[:, 14]
    MTRANS = dados[:, 15]
    NObeyesdad = dados[:, 16]

    classes_unique = np.unique(NObeyesdad)
    class_to_int = {cls: i for i, cls in enumerate(classes_unique)}
    y = np.array([class_to_int[c] for c in NObeyesdad], dtype=int)

    X = np.column_stack([
        np.array(Age, dtype=float),
        np.array(Height, dtype=float),
        np.array(Weight, dtype=float)
    ])

    return X, y

X, y = carregar_arff()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def accuracy():
    correct = sum(yt == yp for yt, yp in zip(y_test, y_pred))
    incorrect = len(y_test) - correct
    return correct, incorrect

def f1_score(y_true, y_pred):
    classes = np.unique(y_true)
    f1_scores = []

    for cls in classes:
        tp = sum((yt == cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fp = sum((yt != cls and yp == cls) for yt, yp in zip(y_true, y_pred))
        fn = sum((yt == cls and yp != cls) for yt, yp in zip(y_true, y_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return np.mean(f1_scores)

def plot_pie_graph(y_test, y_pred):
    correct, incorrect = accuracy()
    f1 = f1_score(y_test, y_pred)

    plt.figure(figsize=(6,6), facecolor="none")
    
    plt.pie(
        [correct, incorrect],
        labels=["Acertos", "Erros"],
        autopct="%1.1f%%",
        colors=["#be76d0", "#c4ecff"],
        textprops={"color": "white"}  # labels brancas
    )

    texto = f"Acurácia: {correct / (correct + incorrect):.4f}   |   F1 Score: {f1:.4f}"
    plt.title("Acurácia \n\n" + texto )

    plt.savefig("pie_chart.png", transparent=True)
    plt.show()


nb = NaiveBayes()


nb.fit(X_train, y_train)

inicio_tempo = time.time()
y_pred = nb.predict(X_test)
fim_tempo = time.time()
tempo_execucao = fim_tempo - inicio_tempo
print(f"Tempo de execução: {tempo_execucao:.4f} segundos")


plot_pie_graph(y_test, y_pred)

