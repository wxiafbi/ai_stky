a1 = [{'a': 'acc1'}, {'b': 'acc1'}, {
    'c': 'acc1'}, {'d': 'acc1'}, {'e': 'acc1'}]
a2 = [{'a': 'acc2'}, {'b': 'acc2'}, {
    'c': 'aBC2'}, {'d': 'acc2'}, {'e': 'acc2'}]
a3 = [{'a': 'acc3'}, {'b': 'acc3'}, {
    'c': 'acc3'}, {'d': 'acc3'}, {'e': 'acc3'}]
a4 = [{'a': 'acc4'}, {'b': 'acc4'}, {
    'c': 'acc4'}, {'d': 'acc4'}, {'e': 'acc5'}]
key_1=['a','b','c','d','e']
tatal = []
tatal.append(a1)
tatal.append(a2)
tatal.append(a3)
tatal.append(a4)
print(tatal)
print(tatal[0][0][key_1[0]])
print(tatal[0])
print(len(tatal[0]))