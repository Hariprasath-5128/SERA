"""
Analyze actual token statistics of raw_chunks vs super_nodes from ChromaDB
to compute the correct equal-budget k pairs.
"""
import sys, statistics
sys.path.insert(0, 'C:/Projects/SERA')
import chromadb

client = chromadb.PersistentClient(path='C:/Projects/SERA/data/chroma')

# ── Fetch raw chunks ──────────────────────────────────────────────
raw_col = client.get_collection('raw_chunks')
raw_total = raw_col.count()
print(f"\n=== RAW CHUNKS ===")
print(f"  Total in DB: {raw_total}")

# Sample up to 2000 to get stable stats
raw_data = raw_col.get(limit=2000, include=['documents'])
raw_docs = raw_data['documents']
raw_word_counts = [len(d.split()) for d in raw_docs if d]
raw_char_counts = [len(d) for d in raw_docs if d]

avg_words_chunk = statistics.mean(raw_word_counts)
med_words_chunk = statistics.median(raw_word_counts)
avg_chars_chunk = statistics.mean(raw_char_counts)
# Token estimate: words ≈ tokens for English; chars/4 as secondary
avg_tok_chunk_words = avg_words_chunk
avg_tok_chunk_chars = avg_chars_chunk / 4

print(f"  Sample size       : {len(raw_docs)}")
print(f"  Avg words/chunk   : {avg_words_chunk:.1f}")
print(f"  Median words/chunk: {med_words_chunk:.1f}")
print(f"  Avg chars/chunk   : {avg_chars_chunk:.1f}")
print(f"  Avg tokens/chunk  : ~{avg_tok_chunk_words:.1f}  (word-based)")
print(f"  Avg tokens/chunk  : ~{avg_tok_chunk_chars:.1f}  (chars/4)")
print(f"  Min words         : {min(raw_word_counts)}")
print(f"  Max words         : {max(raw_word_counts)}")

# ── Fetch super nodes ─────────────────────────────────────────────
sn_col = client.get_collection('super_nodes')
sn_total = sn_col.count()
print(f"\n=== SUPER NODES ===")
print(f"  Total in DB: {sn_total}")

sn_data = sn_col.get(limit=sn_total, include=['documents'])
sn_docs = sn_data['documents']
sn_word_counts = [len(d.split()) for d in sn_docs if d]
sn_char_counts = [len(d) for d in sn_docs if d]

avg_words_sn = statistics.mean(sn_word_counts)
med_words_sn = statistics.median(sn_word_counts)
avg_chars_sn = statistics.mean(sn_char_counts)
avg_tok_sn_words = avg_words_sn
avg_tok_sn_chars = avg_chars_sn / 4

print(f"  Sample size         : {len(sn_docs)}")
print(f"  Avg words/super node: {avg_words_sn:.1f}")
print(f"  Median words/SN     : {med_words_sn:.1f}")
print(f"  Avg chars/super node: {avg_chars_sn:.1f}")
print(f"  Avg tokens/SN       : ~{avg_tok_sn_words:.1f}  (word-based)")
print(f"  Avg tokens/SN       : ~{avg_tok_sn_chars:.1f}  (chars/4)")
print(f"  Min words           : {min(sn_word_counts)}")
print(f"  Max words           : {max(sn_word_counts)}")

# ── Ratio and correct pairs ───────────────────────────────────────
ratio = avg_words_sn / avg_words_chunk
print(f"\n=== TOKEN RATIO ===")
print(f"  Avg words per raw chunk : {avg_words_chunk:.1f}")
print(f"  Avg words per super node: {avg_words_sn:.1f}")
print(f"  Ratio (SN / chunk)      : {ratio:.2f}x")
print(f"  => 1 Super Node ~= {ratio:.1f} raw chunks in word/token volume")

print(f"\n=== CORRECT EQUAL-BUDGET PAIRS ===")
print(f"  (static_k, dynamic_k) -- chosen so static_k * {avg_words_chunk:.0f} words ~= dynamic_k * {avg_words_sn:.0f} words")
print(f"  {'Pair':<12} {'Stat k':>6} {'Dyn k':>6} {'Stat Tokens':>13} {'Dyn Tokens':>12} {'Diff%':>7} {'Label':<12}")
print(f"  {'-'*75}")
for dk in [1, 2, 3, 4, 5]:
    target = dk * avg_words_sn
    sk = round(target / avg_words_chunk)
    if sk == 0:
        sk = 1
    s_tok = sk * avg_words_chunk
    d_tok = dk * avg_words_sn
    diff = abs(s_tok - d_tok) / d_tok * 100
    label = f"s{sk}_d{dk}"
    print(f"  Dynamic k={dk}   {sk:>6}   {dk:>6}   {s_tok:>10.0f} w   {d_tok:>10.0f} w   {diff:>5.1f}%   {label}")
