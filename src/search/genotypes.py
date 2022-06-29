#this is the original_space v0 with all homophilic operations, ref. SANE
NA_PRIMITIVES_v0 = [
  'sage',
  'sage_sum',
  'sage_max',
  'gcn',
  'gin',
  'gat',
  'gat_sym',
  'gat_cos',
  'gat_linear',
  'gat_generalized_linear',
  'geniepath',
]

#this is the proposed heterophilic search space v1, enriching all homo & hetero operations, the proposed AutoHeG used
NA_PRIMITIVES_v1 = [
  'gcnii',
  'cheb',
  #'sign',
  'appnp',
  'fagcn',
  'gprgnn',
  'sgc',
  'supergat',
  'sage',
  'sage_sum',
  'sage_max',
  'gcn',
  'gin',
  'gat',
  'gat_sym',
  'gat_cos',
  'gat_linear',
  'gat_generalized_linear',
  'geniepath',
]

#this is the subset of v1 for illustrating the neccessity of space shrinking with both suset of homo & hetero
NA_PRIMITIVES_v2 = [
  'gcnii',
  'cheb',
  'appnp',
  'fagcn',
  'sage',
  'gat',# followings are added.
  'gprgnn',
  'sgc',
  'supergat'
]

#this is the subset of v1 for illustrating the neccessity of space shrinking, but only with hetero
NA_PRIMITIVES_v3 = [
  'gcnii',
  'cheb',
  'appnp',
  'fagcn',
  'gprgnn',
  'sgc',
  'supergat'
]

SC_PRIMITIVES=[
  'none',
  'skip',
]

'''
SC_PRIMITIVES=[
  #'none',
  'skip',
]
'''
LA_PRIMITIVES=[
  'l_max',
  'l_concat',
  'l_lstm'
]


