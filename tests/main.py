import h5py
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import floyd_warshall
from igraph import Graph
import time
import csv
# Open the HDF5 file
def read_hdf5(arquivo):
    nome_csv = 'motifs_app.csv'
    with open(nome_csv, mode='a', newline='') as arquivo_csv:
        csv_writer = csv.writer(arquivo_csv)
        """ string_motif_3 = [f"motif_3_{i}" for i in range(4)]
        string_motif_4 = [f"motif_4_{i}" for i in range(11)]
        string_motif_5 = [f"motif_5_{i}" for i in range(34)]
        csv_writer.writerow(["exam_id", "lead"] + 
                             string_motif_3 + string_motif_4 + string_motif_5 +  
                             ["duration_motif_3", "duration_motif_4", "duration_motif_5", "array_p_4", "array_p_5"]) """
        with h5py.File(arquivo, 'r') as f:
    # Iterate through top-level groups and their indices
            count = 0
            """ p_array_motif= [([0.3, 0.3, 0.3, 0.3], [0.3, 0.3, 0.3, 0.3, 0.3]),
                            ([0.05, 0.1, 0.15, 0.2],  [0.05, 0.1, 0.15, 0.2, 0.25]),
                            ([0.2, 0.15, 0.1, 0.05],[0.25, 0.2, 0.15, 0.1, 0.05]),
                            ([0, 0.1, 0.18, 0.24], [0, 0.1, 0.18, 0.24, 0.3]),
                            ([0.24, 0.18, 0.1, 0], [0.3, 0.24, 0.18, 0.1, 0] ),
                            ([0.5, 0.25, 0.12, 0.06], [0.5, 0.25, 0.12, 0.06, 0.03]),
                            ([0.06, 0.12, 0.25, 0.5], [0.03, 0.06, 0.12, 0.25, 0.5] ),
                            ] """
            for index, (exam_id, top_group) in enumerate(f.items()):
                if(count >= 1):
                    exit
                print(f'Top-Level Group: {exam_id}')
                p_4 = [0.1, 0.1, 0, 0.1]
                p_5 = [0.1, 0.1, 0, 0.1, 0.1]
                # Iterate through subgroups within the top-level group
                for subgroup_name, subgroup in top_group.items():
                    if(count >= 1):
                        exit
                    print(f'\tSubgroup: {subgroup_name}')                 
                    if 'grafo' in subgroup:
                        grafo_group = subgroup['grafo']
                        G_igraph = Graph.TupleList(zip(grafo_group['row'][()], grafo_group['col'][()]), directed=False)
                        start_time_motif_3 = time.time()
                        print("motifs size 3")
                        count_motif_size_3 =  G_igraph.motifs_randesu(3)
                        end_time_motif_3 = time.time()
                        duration_motif_3 = end_time_motif_3 - start_time_motif_3
                        print(f"duration motif 3: {duration_motif_3}")
                        start_time_motif_4 = time.time()
                        print("motifs size 4")
                        count_motif_size_4 = G_igraph.motifs_randesu(4, p_4)
                        end_time_motif_4 = time.time()
                        duration_motif_4 = end_time_motif_4 - start_time_motif_4
                        print(f"duration motif 4: {duration_motif_4}")
                        start_time_motif_5 = time.time()
                        print("motifs size 5")
                        count_motif_size_5 = G_igraph.motifs_randesu(5, p_5)
                        end_time_motif_5 = time.time()
                        duration_motif_5 = end_time_motif_5 - start_time_motif_5
                        print(f"duration motif 5: {duration_motif_5}")
                        csv_writer.writerow([exam_id, subgroup_name] + 
                                             count_motif_size_3 + count_motif_size_4 + count_motif_size_5 + [duration_motif_3, duration_motif_4, duration_motif_5] + [p_4] + [p_5])
                    count+=1

path_file = "/scratch/raularaju/visibility_graph_classifier/visibility_graphs.hdf5"   
read_hdf5(path_file)                   


""" 
contante:
[0.3, 0.3, 0.3, 0.7]

Linear:
[0.7, 0.8, 0.9, 1]
[1, 0.9, 0.8, 0.7]

Logaritmica (log10): 
[0.7, 0.78, 0.85, 0.9]
[0.9, 0.85, 0.78, 0.7]

Exponencial (1/2):
[1 , 0.5, 0.25, 0.125]
[0.0625, 0.125, 0.25, 0.5] """


""" 
contante:
[0.7, 0.7, 0.7, 0.7, 0.7]

Linear:
[0.5, 0.6, 0.7, 0.8, 0.9]
[0.9, 0.8, 0.7, 0.6, 0.5]

Logaritmica (log10): 
[0.7, 0.78, 0.85, 0.9, 0.95]
[0.95, 0.9, 0.85, 0.78, 0.7]

Exonencial (1/2):
[1, 0.5, 0.25, 0.125, 0.0625]
[0.0625, 0.125, 0.25, 0.5, 1] 

"""


