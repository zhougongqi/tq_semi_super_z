    from sklearn.cluster import KMeans, MiniBatchKMeans
    """
    ######### K MEANS ################################################################
    """
    num_clusters = 5
    km = MiniBatchKMeans(
        n_clusters=num_clusters,
        max_iter=300,
        n_init=10,
        init="k-means++",
        batch_size=100,
        compute_labels=True,
    )
    result = km.fit_predict(soft)
    soft_label_pred = km.labels_
    centroids = km.cluster_centers_
    inertia = km.inertia_