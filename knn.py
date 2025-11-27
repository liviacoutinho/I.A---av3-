import numpy as np
import math
import arff
import time
import matplotlib.pyplot as plt

class KNN:
    def __init__(self, k=5, task='classification'):
        self.k = k
        self.task = task

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def euclidian_distance(self, x1, x2):
        return np.sqrt(np.sum((x1-x2)**2))
    
    def manhattan_distance(self, x1, x2):
        return np.sum(np.abs(x1 - x2))

    def calculate_prediction(self, x, distance_type='euclidean'):
        #decide se é distância euclidiana ou manhattan
        if distance_type == 'euclidean':
            distances = [self.euclidian_distance(x, x_train) for x_train in self.X_train]
        elif distance_type == 'manhattan':
            distances = [self.manhattan_distance(x, x_train) for x_train in self.X_train]


        k_indices = np.argsort(distances)[:self.k]
        k_neares_labels = [self.y_train[i] for i in k_indices]

        if self.task == "classification":
            unique, counts = np.unique(k_neares_labels, return_counts=True)
            return unique[np.argmax(counts)]
        elif self.task == "regression":
            return np.mean(k_neares_labels)
        else:
            raise ValueError("Tarefa nao definida")

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


    X = np.column_stack([
        Height,
        Weight,
        Age
        
    ])

    y = NObeyesdad

    return X, y

X, y=carregar_arff()

X_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNN(k=3, task="classification")
knn.fit(X_train, y_train)

inicio_tempo = time.time()
y_pred = [knn.calculate_prediction(x) for x in x_test]
fim_tempo = time.time()
 
tempo_execucao = fim_tempo - inicio_tempo
print(f"Tempo de execução: {tempo_execucao:.4f} segundos")
def accuracy():
    correct = sum(yt == yp for yt, yp in zip(y_test, y_pred))
    incorrect = len(y_test) - correct
    return correct, incorrect

def calcular_f1_score(y_test, y_pred):
    classes = np.unique(y_test)
    f1_scores = []

    for cls in classes:
        tp = sum((yt == cls and yp == cls) for yt, yp in zip(y_test, y_pred))
        fp = sum((yt != cls and yp == cls) for yt, yp in zip(y_test, y_pred))
        fn = sum((yt == cls and yp != cls) for yt, yp in zip(y_test, y_pred))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0

        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    return np.mean(f1_scores)


def plot_accuracy_pie(y_test, y_pred):
    correct, incorrect = accuracy()
    f1 = calcular_f1_score(y_test, y_pred)

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

    
plot_accuracy_pie(y_test, y_pred)


