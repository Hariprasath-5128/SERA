# SERA Evaluation Summary (Static vs Dynamic)

## 1. Complex Queries (K-Sensitivity)
*Hypothesis: Dynamic pipelines should outperform Static pipelines on complex queries that require synthesizing multiple distinct medical concepts.*\n\n| k | Static ROUGE-L | Dynamic ROUGE-L | Static BioBERT | Dynamic BioBERT | Static Recall | Dynamic Recall |
|---|---|---|---|---|---|---|
| 1 | 0.1719 | 0.1351 | 0.9605 | 0.9239 | 0.1422 | 0.1131 |
| 2 | 0.1506 | 0.1570 | 0.9577 | 0.9576 | 0.1203 | 0.1345 |
| 3 | 0.1505 | 0.1676 | 0.9583 | 0.9629 | 0.1262 | 0.1462 |
| 4 | 0.1566 | 0.1778 | 0.9623 | 0.9672 | 0.1319 | 0.1604 |
| 5 | 0.1647 | 0.1799 | 0.9644 | 0.9682 | 0.1416 | 0.1681 |
| 6 | 0.1657 | 0.1937 | 0.9660 | 0.9701 | 0.1396 | 0.1783 |
| 7 | 0.1691 | 0.1939 | 0.9668 | 0.9704 | 0.1396 | 0.1720 |
| 8 | 0.1717 | 0.1975 | 0.9680 | 0.9702 | 0.1427 | 0.1755 |
| 9 | 0.1733 | 0.1981 | 0.9683 | 0.9710 | 0.1479 | 0.1786 |

## 2. Simple Queries (K-Sensitivity)
| k | Static ROUGE-L | Dynamic ROUGE-L | Static BioBERT | Dynamic BioBERT | Static Recall | Dynamic Recall |
|---|---|---|---|---|---|---|
| 1 | 0.0975 | 0.1152 | 0.9278 | 0.9328 | 0.0762 | 0.0953 |
| 2 | 0.1137 | 0.1232 | 0.9439 | 0.9436 | 0.0898 | 0.1021 |
| 3 | 0.1159 | 0.1269 | 0.9422 | 0.9462 | 0.0935 | 0.1055 |
| 4 | 0.1283 | 0.1284 | 0.9502 | 0.9471 | 0.1069 | 0.1043 |
| 5 | 0.1339 | 0.1285 | 0.9528 | 0.9471 | 0.1129 | 0.1052 |
| 6 | 0.1333 | 0.1268 | 0.9520 | 0.9464 | 0.1137 | 0.1029 |
| 7 | 0.1381 | 0.1218 | 0.9551 | 0.9443 | 0.1206 | 0.0986 |
| 8 | 0.1396 | 0.1254 | 0.9571 | 0.9462 | 0.1234 | 0.1040 |
| 9 | 0.1409 | 0.1225 | 0.9578 | 0.9443 | 0.1225 | 0.1012 |

## 3. Equal Context Budget Comparison
*Comparing Static and Dynamic pipelines when they are constrained to use roughly the same amount of context tokens.*

### Complex Queries
| Budget Pair | Static ROUGE-L | Dynamic ROUGE-L | Static Recall | Dynamic Recall | Static Tokens | Dynamic Tokens |
|---|---|---|---|---|---|---|
| s1_d3 | 0.1710 | 0.1704 | 0.1359 | 0.1464 | 182 | 966 |
| s2_d6 | 0.1563 | 0.1927 | 0.1216 | 0.1716 | 355 | 1925 |
| s3_d9 | 0.1529 | 0.1944 | 0.1265 | 0.1757 | 530 | 2877 |

### Simple Queries
| Budget Pair | Static ROUGE-L | Dynamic ROUGE-L | Static Recall | Dynamic Recall | Static Tokens | Dynamic Tokens |
|---|---|---|---|---|---|---|
| s1_d3 | 0.0939 | 0.1279 | 0.0728 | 0.1059 | 192 | 883 |
| s2_d6 | 0.1111 | 0.1259 | 0.0854 | 0.1041 | 368 | 1803 |
| s3_d9 | 0.1211 | 0.1256 | 0.0963 | 0.1030 | 553 | 2721 |

