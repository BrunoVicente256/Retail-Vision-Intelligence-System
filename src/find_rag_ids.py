"""
Script temporario para descobrir os inspection_ids reais
que o RAG recupera para cada query do ground truth.
Correr: python src/find_rag_ids.py
"""
import sys
sys.path.insert(0, "src")

from rag_memory import query_memoria

queries = [
    "prateleira vazia rutura stock Z_EMPTY",
    "produtos desalinhados desordenados dirty",
    "zona estado critico intervencao imediata",
    "fill rate baixo reposicao urgente",
    "prateleiras bom estado ok abastecidas",
    "ruturas de stock graves severidade alta",
    "violacoes planograma produto fora lugar",
    "prateleira normal com empty parcial subtil",
    "desordem misaligned dirty",
    "fill rate abaixo 70 por cento",
]

print("=" * 65)
print("IDs recuperados pelo RAG por query")
print("=" * 65)

for q in queries:
    r = query_memoria(q, sintetizar=False, k=3)
    print(f"\nQUERY: {q}")
    chunks = r.get("chunks_recuperados", [])
    if not chunks:
        print("  (sem resultados)")
    for c in chunks:
        m = c.get("metadados", {})
        insp_id = m.get("inspection_id", "?")
        zone_id = m.get("zone_id", "?")
        status  = m.get("status", "?")
        rel     = c.get("relevancia", 0)
        print(f"  {insp_id} | {zone_id} | status={status} | rel={rel:.3f}")