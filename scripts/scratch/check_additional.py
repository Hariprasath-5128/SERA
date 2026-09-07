import json

with open(r'C:\Projects\SERA\data\results\eval_20260904_223603\additional_info\all_results.json') as f:
    data = json.load(f)

print('=== STRUCTURE ===')
print(f'Type: {type(data).__name__}')
print(f'Top-level keys: {list(data.keys())}')

for pipeline, records in data.items():
    print(f'\n=== Pipeline: {pipeline} ===')
    print(f'  Total records: {len(records)}')
    if not records:
        print('  [EMPTY - NO DATA]')
        continue

    r0 = records[0]
    print(f'  Keys: {list(r0.keys())}')

    top_ks = sorted(set(r.get("top_k") for r in records))
    print(f'  top_k values used: {top_ks}')

    raiv  = [r["raiv"] for r in records if r.get("raiv") is not None]
    sair  = [r["sair"] for r in records if r.get("sair") is not None]
    uair  = [r["uair"] for r in records if r.get("uair") is not None]
    nonzero_sair = [v for v in sair if v != 0.0]
    nonzero_uair = [v for v in uair if v != 0.0]

    if raiv:
        print(f'  RAIV  -> count={len(raiv)}, avg={sum(raiv)/len(raiv):.2f}, min={min(raiv)}, max={max(raiv)}')
    else:
        print('  RAIV  -> MISSING')

    if nonzero_sair:
        print(f'  SAIR  -> count={len(sair)}, non-zero={len(nonzero_sair)}, avg={sum(nonzero_sair)/len(nonzero_sair):.2f}%')
    else:
        print('  SAIR  -> ALL ZERO or MISSING')

    if nonzero_uair:
        print(f'  UAIR  -> count={len(uair)}, non-zero={len(nonzero_uair)}, avg={sum(nonzero_uair)/len(nonzero_uair):.2f}%')
    else:
        print('  UAIR  -> ALL ZERO or MISSING')

    claims = [r.get("total_claims", 0) for r in records]
    avg_claims = sum(claims)/len(claims) if claims else 0
    zero_claims = sum(1 for c in claims if c == 0)
    print(f'  total_claims -> avg={avg_claims:.2f}, zero_claim_records={zero_claims}/{len(records)}')

    # VALIDITY CHECKS
    print('  --- Validity Checks ---')
    valid = True
    if len(records) == 0:
        print('  [FAIL] No records at all')
        valid = False
    if len(nonzero_sair) == 0:
        print('  [FAIL] SAIR is all zeros - judge never ran')
        valid = False
    if zero_claims == len(records):
        print('  [FAIL] total_claims=0 for all records - judge never ran')
        valid = False
    if not top_ks or top_ks == [None]:
        print('  [FAIL] top_k is missing/null')
        valid = False
    if valid:
        print('  [PASS] Data looks valid')
