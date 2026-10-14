# Validation record — initial 0.1.0

Date: 2026-10-14. Development environment: Python 3.12.14, no production server or real organization documents supplied.

## Completed
- `python -m pytest -q`: 18 tests passed. One dependency deprecation warning from Starlette's httpx TestClient adapter; no test failure.
- `python -m compileall -q app scripts`: passed.
- Chromium headless rendering of real ASGI-generated HTML plus local CSS: chat, document library, administration and account at 375, 768, 1024 and 1440 px; no horizontal document overflow.
- Login and extractive question/answer exercised through ASGI TestClient using synthetic V-101 data.
- Desktop answer and mobile chat screenshots visually inspected internally; not included as deliverables.
- Synthetic PDF two-page references and Word paragraph/table order verified.
- Mocked local and hybrid retrieval/provider checks cover valid/invalid source labels, no-source behavior, no online fallback and HTML escaping.

## Not yet verified
- Actual Ollama inference, online provider compatibility and real E5 embedding inference.
- Real Tesseract OCR / LibreOffice DOC conversion; optional tool availability and error paths are implemented.
- CPU/RAM/latency on the target 16 GB / 5-core server.
- Accuracy, document layout, charts, formulas and operational safety on petrochemical documents.
- Docker image build/start (Docker executable unavailable in this environment).
- Full live-browser HTTP end-to-end navigation; UI snapshots use ASGI response rendering. Functional HTTP paths are covered by TestClient.
- At the initial delivery remote GitHub Actions could not run; the follow-up below records successful publication and CI.

## GitHub publication follow-up

The original target `saeid9013/SIMS-AI` was readable, but authenticated connector writes returned HTTP 403 `Resource not accessible by integration`. No successful write to that original target is claimed.

With the user's authorization, a new public repository `saeid9013/SIMS-AI-RAG` was created using the signed-in browser. Source files, setup, tests, design guidance and Persian documentation are published there in separate web upload commits. Connector writes remain forbidden; browser commits are the successful publication path.

A fresh clone of the new repository at `21ded46b9069ae4cb011dba44a25a846627f1023` passed all 18 tests locally. GitHub Actions run `38633322247` for that commit completed with conclusion `success` on 2026-10-14. Subsequent design/documentation commits do not change executable behavior.

The original local development history remains available in the earlier release bundle; web upload commits have their own history. The empty Python package file created through the editor contains a harmless final newline; the local source copy was aligned to that formatting.

Repository: https://github.com/saeid9013/SIMS-AI-RAG
CI evidence: https://github.com/saeid9013/SIMS-AI-RAG/actions/runs/38633322247
