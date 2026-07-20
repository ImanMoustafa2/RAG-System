# SDS Hybrid-RAG Assistant 🧪

نظام RAG جاهز للعرض (Streamlit) للإجابة على أسئلة حول 4 نشرات سلامة كيميائية
(Ethanol, NaOH, HCl, H2SO4) باستخدام الـ pipeline التالي بالضبط:

```
Chunking      → RecursiveCharacterTextSplitter (size=500, overlap=100)
Embeddings    → BAAI/bge-small-en-v1.5 (نفس الموديل للفهرسة والاستعلام)
Index         → FAISS HNSW  (يتحول لـ IVF-PQ فقط عند نمو الحجم/الذاكرة)
Retrieval     → Hybrid: Dense (FAISS) + BM25 (rank_bm25)
Fusion        → Reciprocal Rank Fusion (RRF, k=60)
Reranker      → Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
Generation    → Groq LLM (llama-3.3-70b-versatile) + prompt يفرض الاستشهاد بالمصدر
Extras        → Metadata filters, Query Rewriting, SQLite cache, JSONL monitoring
```

## هيكل المشروع

```
sds_rag/
├── app.py                  # واجهة Streamlit (Chat / Monitoring / About)
├── requirements.txt
├── data/                   # ملفات الـ SDS الأربعة (PDF)
├── src/
│   ├── config.py            # كل الإعدادات القابلة للتعديل
│   ├── ingestion.py          # تحميل PDF + استخراج الميتاداتا + الـ chunking
│   ├── embeddings.py         # موديل الـ embeddings (singleton)
│   ├── indexing.py           # فهرس FAISS HNSW
│   ├── bm25_index.py         # فهرس BM25
│   ├── retrieval.py          # Hybrid retrieval + RRF + metadata filters
│   ├── reranker.py           # Cross-encoder reranking
│   ├── query_rewriter.py     # إعادة صياغة السؤال قبل البحث
│   ├── generation.py         # بناء البرومبت والاستدعاء عبر Groq
│   ├── cache.py              # كاش SQLite للاستعلامات
│   ├── monitoring.py         # تسجيل زمن كل مرحلة (JSONL)
│   ├── pipeline.py           # الأوركستريتور الذي يجمع كل المراحل
│   └── build_index.py        # سكريبت بناء الفهارس من data/
├── indexes/                 # (يُنشأ تلقائيًا) الفهارس المحفوظة
├── .cache/                  # (يُنشأ تلقائيًا) كاش الاستعلامات
└── logs/                    # (يُنشأ تلقائيًا) سجل المراقبة
```

## التشغيل

```bash
# 1) بيئة افتراضية (اختياري لكن مستحسن)
python3 -m venv venv && source venv/bin/activate

# 2) تثبيت المكتبات
pip install -r requirements.txt

# 3) مفتاح Groq (مجاني من console.groq.com)
export GROQ_API_KEY="your_key_here"

# 4) بناء الفهارس أول مرة (أو من داخل الواجهة بزر "Rebuild indexes")
python -m src.build_index

# 5) تشغيل الواجهة
streamlit run app.py
```

يمكنك أيضًا كتابة مفتاح Groq مباشرة من الشريط الجانبي في الواجهة بدل الـ
environment variable.

## ملاحظات تصميم مهمة (للمناقشة/العرض)

- **لماذا HNSW وليس IVF-PQ؟** الداتا الحالية صغيرة جدًا (عشرات chunks)،
  فـ HNSW يعطي دقة استرجاع شبه كاملة وسرعة عالية بدون خطوة تدريب. الانتقال
  لـ IVF-PQ يكون مبررًا فقط عند نمو الفهرس لملايين المتجهات أو عند ضيق الذاكرة
  (`config.IVF_PQ_SWITCH_THRESHOLD_VECTORS`).
- **لماذا RRF وليس دمج بالـ score مباشرة؟** درجات BM25 والـ cosine similarity
  على مقاييس غير قابلة للمقارنة؛ RRF يعتمد فقط على الترتيب (rank) وليس القيمة
  المطلقة، وبالتالي دمج مستقر بدون الحاجة لـ normalization يدوي.
- **لماذا Cross-Encoder بعد الدمج وليس بدل الاسترجاع الأولي؟** الـ Cross-Encoder
  دقيق جدًا لكنه بطيء (يحسب تفاعل كامل بين السؤال والمقطع)، فهو غير عملي على كل
  الـ corpus؛ لذلك نطبّقه فقط على الـ top-K الناتجة من الدمج.
- **الاستشهاد بالمصادر (citations):** البرومبت في `generation.py` يُلزم النموذج
  بالإشارة إلى `[source_file, section]` بعد كل جملة معلوماتية، ويرفض الإجابة من
  خارج السياق المسترجَع.
- **الكاش (`cache.py`):** SQLite بسيط، مفتاحه hash لـ (السؤال + الفلاتر)، مع TTL
  24 ساعة — يوفر الوقت/التكلفة عند تكرار نفس الأسئلة الشائعة.
- **المراقبة (`monitoring.py`):** كل استعلام يُسجَّل كسطر JSON يحوي زمن كل مرحلة
  (rewrite/retrieval/rerank/generation) وعدد المصادر ونسبة الكاش — تُعرض في تبويب
  "📊 Monitoring" داخل الواجهة.

## توسيع الداتا

لإضافة SDS جديدة، ضع ملف PDF في `data/` ثم اضغط زر "🔄 (Re)build indexes"
من الشريط الجانبي، أو نفّذ `python -m src.build_index` من التيرمينال.
