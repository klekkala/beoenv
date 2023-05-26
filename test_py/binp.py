import binpacking

b = { 'a': 29, 'b': 16, 'c':15, 'd':28, 'e': 21,'f':20, 'g':22, 'h':28, 'i':28, 'j':17 }
bins = binpacking.to_constant_bin_number(b,5)
print("===== dict\n",b,"\n",bins)

b = list(b.values())
bins = binpacking.to_constant_volume(b,36)
print("===== list\n",b,"\n",bins)
