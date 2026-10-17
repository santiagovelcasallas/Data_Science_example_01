"""Compara el AST de las celdas de codigo del notebook anterior con el nuevo. AST igual = solo cambiaron comentarios."""
import ast, json, sys
def celdas(f):
    return [''.join(c['source']) for c in json.load(open(f))['cells'] if c['cell_type'] == 'code']
def norm(src):
    src = '\n'.join(l for l in src.split('\n') if not l.lstrip().startswith(('!', '%')))
    return ast.dump(ast.parse(src))
antes, despues = celdas(sys.argv[1]), celdas(sys.argv[2])
A = [norm(s) for s in antes]; D = [norm(s) for s in despues]
j = 0; cambiadas = []
for i, a in enumerate(A):
    while j < len(D) and D[j] != a:
        j += 1
    if j == len(D):
        cambiadas.append(i); j = 0
    else:
        j += 1
nuevas = [k for k, d in enumerate(D) if d not in A]
com = sum(s.count('\n#') + s.count('# ') for s in despues) - sum(s.count('\n#') + s.count('# ') for s in antes)
print(f"celdas antes: {len(A)} | después: {len(D)}")
print("celdas anteriores con CÓDIGO modificado:", cambiadas or "ninguna")
print("celdas nuevas / con código distinto en la versión nueva:", nuevas or "ninguna")
