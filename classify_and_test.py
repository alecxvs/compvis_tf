import pandas
import io
import pickle
import time
import numpy as np
from sklearn import decomposition, discriminant_analysis
from sklearn.svm import SVC
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist
from functools import reduce


def clock_perf(runtime, msg):
    diff = (time.perf_counter_ns() - runtime) * 1e-6
    print(f" * [{diff:.2f}ms] {msg}")
    runtime = time.perf_counter_ns()


# Based on queryMAP function in MatLab
def query_mAP(train_feat, test_feat, train_ids, test_ids):
    dists = cdist(test_feat, train_feat)  # like pdist2 -- 2D Pairwise Distances in euclidean space

    ap = np.zeros(len(dists))
    for i, row in enumerate(dists):
        top_idx = np.argsort(row)[:10]
        n = reduce(lambda c, label: c + int(label == test_ids[i]), train_ids[top_idx], 0)
        ap[i] = n/10
    return np.mean(ap)


runtime = time.perf_counter_ns()
items = []
print("> Loading features from pickle file...")
with io.open("features.p", "rb") as f:
    while 1:
        try:
            items.append(pickle.load(f))
        except EOFError:
            break

clock_perf(runtime, "Finished")
print("> Splitting training and test data...")

data = pandas.DataFrame.from_records(items, columns=['features', 'label']).sample(frac=1).reset_index(drop=True)
train_length = int(len(data)*5/7)

train_data = np.array(list(data['features'][:train_length]))
test_data = np.array(list(data['features'][train_length:]))
train_labels = np.array(list(data['label'][:train_length]))
test_labels = np.array(list(data['label'][train_length:]))

clock_perf(runtime, "Finished")

print("> Fitting PCA")
pca = decomposition.PCA(32, svd_solver='full')
train_pca = pca.fit_transform(train_data)
test_pca = pca.transform(test_data)
clock_perf(runtime, "Finished")

print("> Fitting LDA")
lda = discriminant_analysis.LinearDiscriminantAnalysis(solver='eigen', n_components=8)
train_lda = lda.fit_transform(train_pca, train_labels)
test_lda = lda.transform(test_pca)
clock_perf(runtime, "Finished")

clock_perf(runtime, f"LDA training accuracy score based on trained classes: {lda.score(train_pca, train_labels)}")
clock_perf(runtime, f"LDA test accuracy score based on trained classes: {lda.score(test_pca, test_labels)}")

mAP = query_mAP(train_lda, test_lda, train_labels, test_labels)
clock_perf(runtime, f"Query mAP performance based on test item vs. top 10 nearest training items: {mAP}")

# Now do SVM classification
# print(f"> Fitting SVM...")
# svm = SVC(gamma="scale").fit(train_data, train_labels)
# clock_perf(runtime, "Finished")

# print("> Measuring accuracy of training data on SVM...")
# svm_train_score = svm.score(train_data, train_labels)
# clock_perf(runtime, f"SVM training accuracy score: {svm_train_score}")

# print("> Measuring accuracy of test data on SVM...")
# svm_test_score = svm.score(test_data, test_labels)
# clock_perf(runtime, f"SVM test accuracy score: {svm_test_score}")

# Now do PCA+LDA+KD-Tree
print("> Building a KD-Tree from LDA data...")
tree = KDTree(train_lda)
clock_perf(runtime, "Finished")
d, indexes = tree.query(test_lda, k=10)

ap = np.zeros(len(indexes))
for i, top_idx in enumerate(indexes):
    n = reduce(lambda c, label: c + int(label == test_labels[i]), train_labels[top_idx], 0)
    ap[i] = n/10
tree_query_map = np.mean(ap)
clock_perf(runtime, f"Query mAP performance of KD-Tree: {tree_query_map}")
