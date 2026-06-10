"""
A/B test: gemini-2.5-flash-lite vs gemini-2.5-flash em casos difíceis.
Compara a deteção visual nos casos onde a verdade foi confirmada por inspeção humana.
"""
import sys
sys.path.insert(0, "src")
sys.stdout.reconfigure(encoding="utf-8")

from shelf_inspector import inspecionar_prateleira

MODELOS = ["gemini-2.5-flash-lite", "gemini-2.5-flash"]

# (caminho, verdade esperada confirmada visualmente)
CASOS = [
    ("images/empty/WhatsApp Image 2026-06-08 at 20.55.52 (2).jpeg",
     "empty parcial: recesso vazio no topo-centro (lixívia). Deve detetar empty_shelf."),
    ("images/empty/WhatsApp Image 2026-06-08 at 20.55.54 (2).jpeg",
     "empty parcial: lacunas dispersas (azeite). Deve detetar empty_shelf."),
    ("images/empty/WhatsApp Image 2026-06-08 at 20.40.23 (1).jpeg",
     "empty parcial (cuidado pessoal). Modelo antigo disse ok/cheia."),
    ("images/normal/WhatsApp Image 2026-06-08 at 20.40.22 (1).jpeg",
     "bolachas COM buracos visíveis. empty_shelf é defensável."),
    ("images/dirty/WhatsApp Image 2026-06-08 at 20.55.52.jpeg",
     "endcap embalagem com caixas semi-vazias. Não é claramente 'damaged'."),
]


def resumo(r):
    if r is None:
        return "  [FALHA — None]"
    tipos = [i.get("type") for i in r.get("issues", [])]
    return (f"  status={r.get('overall_status'):<8} "
            f"fill={r.get('shelf_fill_rate'):.2f}  "
            f"issues={len(tipos)} {tipos}")


for caminho, verdade in CASOS:
    nome = caminho.split("/")[-1]
    print("\n" + "=" * 72)
    print(f"IMAGEM: {nome}")
    print(f"VERDADE: {verdade}")
    print("=" * 72)
    for modelo in MODELOS:
        r = inspecionar_prateleira(caminho, tipo_prompt="cot", modelo=modelo)
        tag = modelo.replace("gemini-2.5-", "")
        print(f"\n[{tag}]")
        print(resumo(r))
        if r:
            print(f"  reasoning: {r.get('model_reasoning', '')[:220]}...")

print("\n" + "=" * 72)
print("A/B test concluído.")
