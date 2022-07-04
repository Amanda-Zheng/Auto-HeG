
import numpy as np
from src.search.genotypes import NA_PRIMITIVES_v0, NA_PRIMITIVES_v1, NA_PRIMITIVES_v2, NA_PRIMITIVES_v3, SC_PRIMITIVES, LA_PRIMITIVES
import logging

def random_geno(NA_PRIMITIVES):
    num_na = 3
    num_sc = 3
    num_la = 1
    gene = []
    for k in range(num_na):
        print(len(NA_PRIMITIVES))
        opt= np.random.randint(0, len(NA_PRIMITIVES))
        gene.append(NA_PRIMITIVES[opt])
    for k in range(num_sc):
        print(len(SC_PRIMITIVES))
        opt= np.random.randint(0, len(SC_PRIMITIVES))
        gene.append(SC_PRIMITIVES[opt])
    for k in range(num_la):
        print(len(LA_PRIMITIVES))
        opt= np.random.randint(0, len(LA_PRIMITIVES))
        gene.append(LA_PRIMITIVES[opt])
    geno_out = '||'.join(gene)
    return geno_out

res = []
for i in range(20):
    logging.info('searched {}-th ...'.format(i + 1))
    seed = np.random.randint(0, 10000)
    genotype_v3 = random_geno(NA_PRIMITIVES_v3)
    print(genotype_v3)
    res.append('seed={},genotype_v3={}'.format(seed, genotype_v3))
filename = 'rand_geno/Random_searched_res_genotype_v3.txt'
fw = open(filename, 'w+')
fw.write('\n'.join(res))
fw.close()