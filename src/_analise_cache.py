"""Diagnóstico temporário: audita a qualidade das inspeções em cache."""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter

sys.stdout.reconfigure(encoding="utf-8")

CACHE = Path("cache/vision")

# O que se espera ver em cada categoria (issue types plausíveis e status)
ESPERADO = {
    "normal":    {"status": {"ok"},                "issues_esperados": set()},
    "empty":     {"status": {"warning", "critical"}, "issues_esperados": {"empty_shelf"}},
    "dirty":     {"status": {"warning", "critical"}, "issues_esperados": {"damaged", "misaligned"}},
    "planogram": {"status": {"warning", "critical"}, "issues_esperados": {"wrong_product", "label_missing", "misaligned"}},
    "ambiguous": {"status": None,                   "issues_esperados": None},  # tolerante
}

registos = []
for f in sorted(CACHE.glob("*.json")):
    if f.name == "dataset_final.json":
        continue
    try:
        d = json.loads(f.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[CORROMPIDO] {f.name}: {e}")
        continue
    img = d.get("image_path", "")
    cat = Path(img).parent.name.lower()
    issues = d.get("issues", [])
    registos.append({
        "ficheiro": f.name,
        "imagem": Path(img).name,
        "categoria": cat,
        "status": d.get("overall_status"),
        "fill": d.get("shelf_fill_rate"),
        "n_issues": len(issues),
        "tipos": [i.get("type") for i in issues],
        "confs": [i.get("confidence") for i in issues],
        "reasoning": d.get("model_reasoning", ""),
        "products": d.get("products_detected", []),
    })

print(f"\n{'='*70}\nTOTAL DE INSPEÇÕES EM CACHE: {len(registos)}\n{'='*70}")

# Distribuição por categoria
por_cat = defaultdict(list)
for r in registos:
    por_cat[r["categoria"]].append(r)

print("\n--- COBERTURA POR CATEGORIA ---")
for cat in ["normal", "empty", "dirty", "ambiguous", "planogram"]:
    print(f"  {cat:<10}: {len(por_cat.get(cat, []))} inspeções")
outras = set(por_cat) - {"normal","empty","dirty","ambiguous","planogram"}
for cat in outras:
    print(f"  [?] {cat:<10}: {len(por_cat[cat])} inspeções")

# Análise de coerência por categoria
print("\n--- COERÊNCIA: status e tipos de issue por categoria ---")
suspeitos = []
for cat in ["normal", "empty", "dirty", "ambiguous", "planogram"]:
    rs = por_cat.get(cat, [])
    if not rs:
        continue
    status_dist = Counter(r["status"] for r in rs)
    tipos_dist = Counter(t for r in rs for t in r["tipos"])
    fills = [r["fill"] for r in rs if isinstance(r["fill"], (int, float))]
    fill_med = sum(fills)/len(fills) if fills else 0
    print(f"\n  [{cat.upper()}]  n={len(rs)}  fill_médio={fill_med:.2f}")
    print(f"     status : {dict(status_dist)}")
    print(f"     tipos  : {dict(tipos_dist)}")

    exp = ESPERADO[cat]
    for r in rs:
        motivos = []
        # status fora do esperado
        if exp["status"] and r["status"] not in exp["status"]:
            motivos.append(f"status={r['status']} (esperado {exp['status']})")
        # nenhum issue esperado presente, mas categoria problemática
        if exp["issues_esperados"]:
            tem_esperado = any(t in exp["issues_esperados"] for t in r["tipos"])
            if not tem_esperado:
                motivos.append(f"sem issue esperado; viu={r['tipos'] or 'nenhum'}")
        # normal com issues
        if cat == "normal" and r["n_issues"] > 0:
            motivos.append(f"normal mas {r['n_issues']} issue(s): {r['tipos']}")
        if motivos:
            suspeitos.append((cat, r, motivos))

print(f"\n{'='*70}\nINSPEÇÕES SUSPEITAS (potenciais erros de classificação): {len(suspeitos)}\n{'='*70}")
for cat, r, motivos in suspeitos:
    print(f"\n  [{cat}] {r['imagem']}  (status={r['status']}, fill={r['fill']})")
    for m in motivos:
        print(f"      >> {m}")
    print(f"      reasoning: {r['reasoning'][:160]}...")

print(f"\n{'='*70}\nINDICADORES DE CACHE OBSOLETA\n{'='*70}")
fora_escala = [r for r in registos if isinstance(r["fill"], (int, float)) and r["fill"] > 1.0]
print(f"  Inspeções com fill_rate > 1.0 (escala 0-100, código ANTIGO): {len(fora_escala)}/{len(registos)}")
confs_fora = [c for r in registos for c in r["confs"] if isinstance(c, (int, float)) and c > 1.0]
print(f"  Confidences > 1.0 (escala 0-100, código ANTIGO): {len(confs_fora)}")

# Imagens distintas vs entradas de cache 
imgs_distintas = defaultdict(set)
for r in registos:
    imgs_distintas[r["categoria"]].add(r["imagem"])
print("\n  Imagens DISTINTAS cobertas por categoria (vs imagens no disco):")
no_disco = {"normal": 16, "empty": 16, "dirty": 12, "ambiguous": 8, "planogram": 2}
for cat in ["normal", "empty", "dirty", "ambiguous", "planogram"]:
    print(f"    {cat:<10}: {len(imgs_distintas.get(cat, set())):>2} distintas / {no_disco[cat]} no disco")

# Categoria "?" — image_path que não mapeia a pasta conhecida
print("\n  Entradas com categoria não reconhecida (image_path):")
for r in registos:
    if r["categoria"] not in ("normal","empty","dirty","ambiguous","planogram"):
        print(f"    {r['ficheiro']}: path='{r['imagem']}' cat='{r['categoria']}'")

# A FALHA MAIS GRAVE: prateleira EMPTY classificada como OK
print(f"\n{'='*70}\nFALHA CRÍTICA: imagens EMPTY classificadas como 'ok'\n{'='*70}")
for r in por_cat.get("empty", []):
    if r["status"] == "ok":
        print(f"  {r['imagem']}  fill={r['fill']}  issues={r['tipos'] or 'NENHUM'}")
        print(f"     reasoning: {r['reasoning'][:140]}...")

# Qualidade do model_reasoning
print(f"\n{'='*70}\nQUALIDADE DO model_reasoning\n{'='*70}")
vazios = [r for r in registos if len(r["reasoning"].strip()) < 40]
curtos = [r for r in registos if 40 <= len(r["reasoning"].strip()) < 120]
print(f"  Reasoning quase-vazio (<40 chars): {len(vazios)}")
print(f"  Reasoning curto (40-120 chars)   : {len(curtos)}")
comprimentos = [len(r["reasoning"]) for r in registos]
if comprimentos:
    print(f"  Comprimento médio: {sum(comprimentos)/len(comprimentos):.0f} chars")
    print(f"  Min/Max: {min(comprimentos)}/{max(comprimentos)} chars")

# Confidence
print(f"\n--- CONFIDENCE dos issues ---")
todas_confs = [c for r in registos for c in r["confs"] if isinstance(c, (int, float))]
if todas_confs:
    print(f"  n issues com confidence: {len(todas_confs)}")
    print(f"  média: {sum(todas_confs)/len(todas_confs):.2f}")
    print(f"  zeros (confidence=0.0): {sum(1 for c in todas_confs if c == 0.0)}")
