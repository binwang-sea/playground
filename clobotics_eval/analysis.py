import json
p = '/Users/bin.wang/Projects/jets_clobotics/response.json'

data = json.load(open(p))
products = data.get('data', {}).get('result', {}).get('products', [])
ident_set = {p.get('identfier') for p in products if p.get('identfier')}
print(ident_set)
print(len(ident_set))

for item in ident_set:
    print(item)