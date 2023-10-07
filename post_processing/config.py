"""
    Script for the parametrization of each of the used algorithms 
"""
class DBSCAN_parameters:
        """
            Hyperparameter definition for the DBSCAN algorithm 
        """
        proba = 0.75
        min_samples = 10
        eps = 0.01
        cluster_size = None
        cluster_eps = None       

        def __init__(self) -> None:
                pass 

class DBSCAN_lowres(DBSCAN_parameters):
        """
            DBSCAN parameters for the low resoliton protcol (2018/2019)
        """
        def __init__(self) -> None:
                super().__init__()

class DBSCAN_highres(DBSCAN_parameters):
        """
            DBSCAN parameters for the high resoliton protcol (2019)
        """
        min_samples = 15

        def __init__(self) -> None:
                super().__init__()

