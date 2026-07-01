# Literature Survey: RAG for WISE 2026

**Date:** 2026-06-11
**Topic:** RAG for Web Information Retrieval & Text Analytics
**Venue:** WISE 2026 (Web Information Systems Engineering)

---

## Executive Summary

Based on a comprehensive literature survey of recent RAG (Retrieval-Augmented Generation) research from 2023-2025, we identify several promising directions aligned with WISE 2026's "Web Information Retrieval and Text Analytics" track. The field has rapidly evolved from basic RAG architectures to sophisticated systems with self-reflection, hierarchical organization, and corrective mechanisms.

**Key Finding:** The most promising research gap for WISE 2026 is in **Web-scale adaptive RAG systems** that can dynamically handle:
1. Dynamic/real-time web content
2. Multi-modal web documents (images + text)
3. Long-context web documents (PDFs, articles)
4. Query-adaptive retrieval strategies

---

## Research Landscape Map

### 1. **Foundational RAG Architecture** (Mature)

| Component | Status | Key Papers |
|-----------|--------|------------|
| Basic RAG Pipeline | Well-established | Lewis et al. (2020), Karpukhin et al. (2020) |
| Dense Retrieval | Mature | ColBERT v2, DPR, Contriever |
| Hybrid Search | Production-ready | SPLADE, DeepImpact |
| Re-ranking | Well-studied | BGE-reranker, Cross-encoders |

**Assessment:** Core RAG technology is mature. Incremental improvements in this area are less likely to be novel enough for a research venue.

---

### 2. **Advanced RAG Techniques** (Active Research Area)

#### Self-Reflective RAG
- **Self-RAG** (Asai et al., arXiv:2310.11511)
  - Trains LM to retrieve, generate, AND critique using reflection tokens
  - On-demand retrieval (not fixed k documents)
  - Outperforms ChatGPT on open-domain QA, reasoning, fact verification

#### Corrective RAG
- **CRAG** (arXiv:2401.15884, Jan 2024)
  - Evaluates retrieved document quality before generation
  - Three operational modes based on retrieval confidence
  - Knowledge refinement process

#### Hierarchical RAG
- **RAPTOR** (Stanford, 2024)
  - Recursive abstractive processing for tree-organized retrieval
  - Hierarchical clustering of document chunks
  - Multi-level abstraction for complex queries

**Assessment:** These approaches represent the cutting edge (2023-2024), but several papers already exist. Need to find a specific angle within these themes.

---

### 3. **Web-Specific RAG** (Research Gap ⭐)

| Direction | Status | Opportunity |
|-----------|--------|------------|
| **Dynamic Web RAG** | Limited research | Real-time web content handling |
| **Web-scale RAG** | Early research | Distributed retrieval for billions of docs |
| **Multi-modal Web RAG** | Emerging | Image-text-webpage understanding |
| **Long Document RAG** | Growing | PDF/article-level retrieval |
| **Web Crawler + RAG** | Under-explored | End-to-end web pipeline |

**Key Insight:** Most RAG research focuses on static document collections (Wikipedia, corpora). Web-specific challenges (dynamic content, HTML structure, multi-modal pages, scale) are under-explored.

---

### 4. **Evaluation & Benchmarks** (Mature but evolving)

| Benchmark | Focus | Relevance |
|-----------|-------|-----------|
| **RAGAS** | Faithfulness, answer relevance | General RAG evaluation |
| **ARES** | 2024 benchmark for RAG | Comprehensive metrics |
| **Natural Questions, MS MARCO** | Open-domain QA | Established baselines |
| **Web-specific benchmarks** | *Limited* | **Research opportunity** |

**Gap:** No comprehensive benchmark for Web-specific RAG tasks (crawling + retrieval + generation).

---

## Key Papers Summary

### Survey Papers (2024)

1. **[arXiv:2401.05809]** "A Survey on Retrieval-Augmented Generation for Large Language Models" (Jan 2024)
   - Comprehensive overview of RAG architectures
   - Covers retrieval and generation component advances

2. **[arXiv:2402.01342]** "Retrieval-Augmented Generation: A Comprehensive Survey of Methods and Applications" (Feb 2024)
   - Systematic review: modular, end-to-end, hybrid RAG
   - Benchmarks and evaluation metrics

3. **[arXiv:2312.10997]** "Retrieval-Augmented Generation for Large Language Models: A Survey" (Dec 2023)
   - Paradigms, foundations, applications
   - Categorization of RAG frameworks

### Technical Advances (2023-2024)

4. **Self-RAG** (arXiv:2310.11511, Oct 2023)
   - **Problem:** Fixed-k retrieval reduces LM versatility
   - **Solution:** On-demand retrieval + reflection tokens
   - **Results:** Outperforms ChatGPT and Llama2-chat

5. **CRAG** (arXiv:2401.15884, Jan 2024)
   - **Problem:** Retrieved documents not always reliable
   - **Solution:** Evaluate and refine retrieved documents
   - **Method:** Three operational modes + knowledge refinement

6. **RAPTOR** (Stanford, 2024)
   - **Problem:** Flat chunking misses hierarchical relationships
   - **Solution:** Tree-organized retrieval via recursive clustering
   - **Results:** Improved multi-hop reasoning and broad document understanding

### HyDE (2022)

7. **HyDE** (arXiv:2212.08075)
   - Generates hypothetical documents from queries
   - Uses hypothetical embeddings for retrieval
   - Improves dense retrieval performance

---

## Open Problems & Research Gaps

### High-Opportunity Gaps for WISE 2026

#### 1. **Dynamic Web RAG** ⭐⭐⭐
**Problem:** Existing RAG assumes static document collections. Web content changes continuously.

**Research Questions:**
- How to handle temporal decay in retrieved web content?
- How to efficiently update vector indexes for dynamic web pages?
- How to detect and handle outdated information in retrieved documents?

**Why WISE:** Directly addresses "Web Information Retrieval" and "Web Data Models"

#### 2. **Multi-Modal Web RAG** ⭐⭐⭐
**Problem:** Web pages are inherently multi-modal (text, images, layouts, links). Most RAG only handles text.

**Research Questions:**
- How to retrieve from multi-modal web documents (screenshots, HTML, text)?
- How to incorporate visual layout information in retrieval?
- How to handle image-text queries on web content?

**Why WISE:** Fits "Rich Web UI and HCI" + "Web Information Retrieval"

#### 3. **Long-Context Web Documents** ⭐⭐
**Problem:** Web documents (PDFs, long articles) exceed typical chunk limits. Important info gets lost.

**Research Questions:**
- How to effectively retrieve from very long web documents?
- Can hierarchical retrieval (like RAPTOR) be adapted for web-scale?
- How to maintain coherence across retrieved chunks from long documents?

**Why WISE:** Addresses "Web Information Quality and Fusion"

#### 4. **Web-Scale Distributed RAG** ⭐⭐
**Problem:** RAG at billion-document scale presents engineering and algorithmic challenges.

**Research Questions:**
- How to efficiently distribute retrieval across many nodes?
- How to balance latency vs. accuracy at web scale?
- What architectures work best for distributed RAG?

**Why WISE:** Fits "Web-scale Distributed and Cloud Computing" + "Web Big Data Techniques"

#### 5. **Adaptive Query Routing for Web RAG** ⭐⭐
**Problem:** Different queries need different retrieval strategies (keyword vs. semantic vs. hybrid).

**Research Questions:**
- How to classify queries and route to appropriate retrieval?
- Can we learn optimal retrieval strategies per query type?
- How to handle multi-step queries that need iterative retrieval?

**Why WISE:** Addresses "Web Agents and Web Intelligence" + "LLM Web-based applications"

---

## Methodological Trends

### What Works (Established Techniques)
- ✅ Dense retrieval (sentence transformers, ColBERT)
- ✅ Hybrid search (dense + sparse)
- ✅ Cross-encoder re-ranking
- ✅ Query expansion and rewriting
- ✅ Multi-stage retrieval pipelines

### What's Emerging (2024)
- 🔜 Self-reflective mechanisms (Self-RAG style)
- 🔜 Corrective loops (CRAG style)
- 🔜 Hierarchical organization (RAPTOR style)
- 🔜 On-demand retrieval (not fixed k)
- 🔜 Multi-modal retrieval

### What's Missing (Opportunities)
- ❌ Web-specific temporal handling
- ❌ End-to-end crawler + RAG pipelines
- ❌ Visual + textual web understanding
- ❌ Query-adaptive retrieval for web
- ❌ Evaluation benchmarks for web RAG

---

## WISE 2026 Alignment Analysis

### WISE Topics of Interest
Our research directions align with these WISE 2026 topics:

| WISE Topic | Our Direction | Fit |
|------------|----------------|-----|
| **Web Information Retrieval and Text Analytics** ⭐⭐ | All RAG directions | ✅ Perfect fit |
| **LLM Web-based applications** | Web-specific RAG | ✅ Strong fit |
| **Web Mining** | Web-scale RAG | ✅ Strong fit |
| **Web Big Data Techniques and Mining** | Distributed RAG | ✅ Good fit |
| **Rich Web UI and HCI** | Multi-modal RAG | ✅ Good fit |
| **Web Agents and Web Intelligence** | Adaptive RAG | ✅ Good fit |
| **Web Information Quality and Fusion** | Dynamic RAG | ✅ Good fit |

**Conclusion:** RAG for Web Information Retrieval is an **excellent fit** for WISE 2026's themes.

---

## Recommended Research Directions (Ranked)

### 🥇 **Direction 1: Dynamic Web RAG with Temporal Awareness**
**Novelty:** High - limited existing work
**Feasibility:** High - can build on existing RAG frameworks
**WISE Fit:** Perfect - directly addresses Web IR

### 🥈 **Direction 2: Multi-Modal RAG for Web Page Understanding**
**Novelty:** High - emerging area
**Feasibility:** Medium - requires vision-language models
**WISE Fit:** Strong - multi-modal web intelligence

### 🥉 **Direction 3: Long-Context RAPTOR for Web Documents**
**Novelty:** Medium - extends existing RAPTOR work
**Feasibility:** High - RAPTOR code is available
**WISE Fit:** Good - web document understanding

---

## Next Steps

1. **Idea Generation** → Use `/idea-creator` to brainstorm specific research questions within top directions
2. **Novelty Check** → Verify gaps aren't already filled by concurrent work
3. **Pilot Experiments** → Test feasibility with small-scale implementations

---

## Sources

- arXiv: [2310.11511] Self-RAG
- arXiv: [2401.15884] CRAG
- arXiv: [2401.05809] RAG Survey
- arXiv: [2402.01342] Comprehensive RAG Survey
- arXiv: [2312.10997] RAG for LLMs Survey
- RAPTOR (Stanford, 2024)
- HyDE (arXiv:2212.08075)
- Web Search Results (2024-2026)
